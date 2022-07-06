 function MI_CoAdaptiveLearning(recordingFolder)
%% MI Online Scaffolding
% This code creates an online EEG buffer which utilizes the model trained
% offline, and corresponding conditions, to classify between the possible labels.
% Furthermore, this code adds an "online learning" phase in which the
% subject is shown a specific label which she/he should imagine. After a
% defined amount of labeled trials, the classifier is updated.

% Assuming: 
% 1. EEG is recorded using openBCI and streamed through LSL.
% 2. A preliminary MI classifier has been trained.
% 3. A different machine/client is reading this LSL oulet stream for the commands sent through this code
% 4. Target labels are [-1 0 1] (left idle right)

% 1. Add a "voting machine" which takes the classification and counts how
% many consecutive answers in the same direction / target to get a high(er)
% accuracy rate, even though it slows down the process by a large factor.
% 2. Add an online learn-with-feedback mechanism where there is a visual feedback to
% one side (or idle) with a confidence bar showing the classification being made.
% 3. Advanced = add an online reinforcement code that updates the
% classifier with the wrong & right class classifications.
% 4. Add a serial classifier which predicts attention levels and updates
% the classifier only if "focus" is above a certain threshold.

%% This code is part of the BCI-4-ALS Course written by Asaf Harel
% (harelasa@post.bgu.ac.il) in 2021. You are free to use, change, adapt and
% so on - but please cite properly if published.
% This code was edited by HUJI Team 35 as part of the BCI-4-ALS course.
% NOTE: THIS CODE STILL NEEDS TO BE OPTIMIZED.

close all
clc

%% Addpath for relevant folders - original recording folder and LSL folders
addpath(genpath('C:\Toolboxes\LabStreamingLayer\'))       
%% Set params - %add to different function/file returns param.struct
params = set_params();
relevantFeatures = load([rootRecordingFolder '\AllDataTopFeaturesIdx.mat']).AllDataTopFeaturesIdx;             % load best features from extraction & selection stage
load([rootRecordingFolder '\EEG_chans.mat']);                % load EEG channel names that were used
load([rootRecordingFolder '\CSP_flag.mat']);
load([recordingFolder '\wCSPTrainLR.mat']);                   % load weights of CSP left vs right
load([recordingFolder '\wCSPTrainLI.mat']);                   % load weights of CSP left vs right
load([recordingFolder '\wCSPTrainRI.mat']);                   % load weights of CSP left vs right
load(strcat(recordingFolder, 'trainedModel.mat'));            % load model weights from offline section - assuming trained matlab incrementalMulticlassModel


Fs = 125; 
% Load cue images
images{1} = imread(params.leftImageName, 'jpeg');
images{2} = imread(params.squareImageName, 'jpeg');
images{3} = imread(params.rightImageName, 'jpeg');
% prepare the cue vector
numTotalTrials = params.numTrials * params.numConditions;
cueVec = zeros(1, numTotalTrials);
learnTrials = zeros(1, numTotalTrials); 
testTrials = zeros(1, numTotalTrials);
counter = 1;
for ii=1:params.numRep
    % add trials for learning
    cueVecTemp = prepareTraining(params.numTrialsLearnRep, params.numConditions);
    cueVec(counter:counter+length(cueVecTemp)-1) = cueVecTemp; 
    learnTrials(counter:counter+length(cueVecTemp)-1) = 1;
    counter = counter+length(cueVecTemp);
    % add trials for test
    cueVecTemp = prepareTraining(params.numTrialsTestRep, params.numConditions);
    cueVec(counter:counter+length(cueVecTemp)-1) = cueVecTemp;
    testTrials(counter:counter+length(cueVecTemp)-1) = 1;
    counter = counter+length(cueVecTemp);
end

%% Lab Streaming Layer Init
[command_Outlet, EEG_Inlet] = LSL_Init(params);

disp([char(10) 'Open Lab Recorder & check for MarkerStream and EEG stream, start recording, return here and hit any key to continue.']);
pause;                                  % wait for experimenter to press a key

%% Initialize some more variables:
class = {'Left', 'Idle', 'Right'};
myPrediction = [];                                  % predictions vector
myBuffer = [];                                      % buffer matrix
iteration = 0;                                      % iteration counter
decCount = 0;                                       % decision counter

%% 
pause(params.bufferPause);                          % give the system some time to buffer data
myChunk = EEG_Inlet.pull_chunk();                   % get a chunk from the EEG LSL stream to get the buffer going

%% Screen Setup 
monitorPos = get(0,'MonitorPositions'); % monitor position and number of monitors
monitorN = size(monitorPos, 1);
choosenMonitor = 1;                     % which monitor to use TODO: make a parameter                                 
if choosenMonitor < monitorN            % if no 2nd monitor found, use the main monitor
    choosenMonitor = 1;
    disp('Another monitored is not detected, using main monitor.')
end
figurePos = monitorPos(choosenMonitor, :);  % get choosen monitor position
figure('outerPosition',figurePos);          % open full screen monitor
MainFig = gcf;                              % get the figure and axes handles
hAx  = gca;
set(hAx,'Unit','normalized','Position',[0 0 1 1]); % set the axes to full screen
set(MainFig,'menubar','none');              % hide the toolbar   
set(MainFig,'NumberTitle','off');           % hide the title
set(hAx,'color', 'black');                  % set background color
hAx.XLim = [0, 1];                          % lock axes limits
hAx.YLim = [0, 1];
hold on

% axes('Position',[.7 .7 .2 .2])
learn_flag = 0;     % indicates if we are in a learning trial or in a test trial
test_correct = zeros(1, params.numRep);
num_classifications = zeros(1, params.numRep);
counterRep = 0;    % counts the test repeat
%% This is the main online script
for trial = 1:numTotalTrials
    % check if we're in a learning trial or a test trial
    if learnTrials(trial) == 1
       learn_flag = 1;
    else
       learn_flag = 0; 
       if trial == 1 || (testTrials(trial-1) == 0 && testTrials(trial) == 1)
           counterRep = counterRep + 1;
       end
    end
    disp(['starting trial' num2str(trial)])
    command_Outlet.push_sample(params.startTrialMarker)
    currentClass = cueVec(trial);
    % ready cue
    hReady = text(0.5,0.5 , 'Get ready',...
        'HorizontalAlignment', 'Center', 'Color', 'white', 'FontSize', 40);        
    % display target cue
    hCue = image(flip(images{currentClass}, 1),  'XData', [0.25, 0.75],...
        'YData', [0.2, 0.8 * ...
        size(images{currentClass},1) ./ size(images{currentClass},2)]);
    pause(params.readyLength)
    delete(hReady)                                         % Clear ready sign
    hStart = text(0.5,0.5 , 'Imagine',...
        'HorizontalAlignment', 'Center', 'Color', 'white', 'FontSize', 40);  
    %     pause(params.cueLength);                           % Pause for cue length
    
    EEG_Inlet.pull_chunk(); % empty the eeg stream
    trialStart = tic;
    while toc(trialStart) < params.trialTime
        iteration = iteration + 1;                  % count iterations
        
        pause(0.1)
        myChunk = EEG_Inlet.pull_chunk();           % get data from the inlet
        % check if myChunk is empty and print status, also
        % local buffer:
        if ~isempty(myChunk)
            % We want to use all of the electrodes
            myBuffer = [myBuffer myChunk];        % append new data to the current buffer
        else
            fprintf(strcat('Houston, we have a problem. Iteration:',num2str(iteration),' did not have any data.'));
        end
        
        % Check if buffer size exceeds the buffer length
        if (size(myBuffer,2)>(params.bufferLength*Fs))
            decCount = decCount + 1;            % decision counter
            block = [myBuffer];                 % move data to a "block" variable
            
            % Pre-process the data
            block = MI2_preprocess([], 0, block, Fs, EEG_chans);
            % Extract features from the buffered block:
            EEG_Features = MI4_featureExtraction([], CSP_flag, block, relevantFeatures, wCSPTrainLR, wCSPTrainLI, wCSPTrainRI);
            
            % Predict using previously learned model:
            [negloss_combine, myPrediction(decCount)] = trainedModel.predict(EEG_Features);
            if ~learn_flag
               num_classifications(counterRep) = num_classifications(counterRep) + 1; 
            end

            if params.feedbackFlag && learn_flag
                delete(hStart);                
                plotEstimate(negloss_combine, images, hAx); 
                pause(2)
                cla
                axes(hAx)
                cla
                pause(0.3)
                hold on
            end
            fprintf(strcat('Iteration: ', num2str(iteration)));
            fprintf(strcat(' The estimated target is: ', num2str(myPrediction(decCount)), '\n'));
            
            [final_vote] = sendVote(myPrediction(decCount));
            
            % Update classifier - this should be done very gently!
            if final_vote ~= (cueVec(trial)) 
                if learn_flag
                    % TO-DO: we might not want to update model based on
                    % every wrong classification
                    
                    wrongClass(decCount,:,:) = EEG_Features;
                    wrongClassLabel(decCount) = cueVec(trial);

                    trainedModel = OnlineLearn(trainedModel,EEG_Features,currentClass);
                end
            else
                correctClass(decCount,:,:) = EEG_Features;
                correctLabel(decCount) = cueVec(trial);
                % Send command through LSL:
                command_Outlet.push_sample(final_vote);
                if ~learn_flag
                    test_correct(counterRep) = test_correct(counterRep) + 1;
                end
            end
            % clear buffer
            myBuffer = [];
            EEG_Inlet.pull_chunk(); % empty the eeg stream
        end
    end
    cla;
end
% test accuracy
test_accuracy = (test_correct ./ num_classifications) * 100;
fig = figure;
plot_accuracy(test_accuracy, 100/params.numConditions);
savefig(fig, strcat(recordingFolder, '\CAL_accuracy.fig'));
disp('Finished')
command_Outlet.push_sample(params.endTrial);
% save updated trained model
save(strcat(recordingFolder, '\trainedModel_CAL.mat'), 'trainedModel');
disp('!!!!!!! Stop the LabRecorder recording!');
