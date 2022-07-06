function MI_OnlineFeedback(rootRecordingFolder)
%% MI Online
% This code creates an online EEG buffer which utilizes the model trained
% offline, and corresponding conditions, to classify between the possible labels.
% After each trial the subject is shown what the label classified by the
% model.

% Parameter:
% rootRecordingFolder: root folder in which to save folder with online data
%                      if predict_type is "predict", rootRecordingFolder
%                      must have training data


% Assumptions: 
% 1. EEG is recorded using openBCI and streamed through LSL.
% 2. A preliminary MI classifier has been trained.
% 3. A different machine/client is reading this LSL oulet stream for the commands sent through this code
% 4. Target labels are [1, 2, 3] (idle, left right)


%% This code is part of the BCI-4-ALS Course and adapted from code written by Asaf Harel
% (harelasa@post.bgu.ac.il) in 2021. You are free to use, change, adapt and
% so on - but please cite properly if published.
% This code was adapated by HUJI team 35 as part of the BCI-4-ALS course

close all
clc

%% Addpath for relevant folders - original recording folder and LSL folders

global_variables

addpath(genpath(lsl_lib_path))

% Subject and recording parameters:
subID = input('Please enter subject ID/Name: ');    % prompt to enter subject ID or name
if nargin < 1
    rootRecordingFolder = 'C:\Recordings\';                      % define recording folder location
end

% Define recording folder location and create the folder:
recordingFolder = strcat(rootRecordingFolder,'\Sub',num2str(subID),'\');
mkdir(recordingFolder);


%% Load model related data - relevantFeatures and trainedModel
num_models = "1"; % 1 - multiclass classifier, 2 - double binary classifier
predict_type = 'committee_predict'; % committee_predict - majority vote of multiple models / "predict" - vote of model trained on rootRecordingFolder
if strcmp(predict_type, "predict")
    relevantFeatures = load([rootRecordingFolder '\AllDataTopFeaturesIdx.mat']).AllDataTopFeaturesIdx;         % load best features from extraction & selection stage
    load([rootRecordingFolder '\EEG_chans.mat']);                % load EEG channel names that were used
    load([rootRecordingFolder '\CSP_flag.mat']);
    if CSP_flag
        load([rootRecordingFolder '\wCSPTrainLR.mat']);                   % load weights of CSP left vs right
        load([rootRecordingFolder '\wCSPTrainLI.mat']);                   % load weights of CSP left vs idle
        load([rootRecordingFolder '\wCSPTrainRI.mat']);                   % load weights of CSP right vs idle
    else
       wCSPTrainLR = []; wCSPTrainLI = []; wCSPTrainRI = [];
    end
    % train model on rootRecordingFolder
    pyrunfile("MI5_classifier.py", featuresVariable='AllDataTopFeatures', recfolder=rootRecordingFolder, action="train", num_models=num_models);  
    recordingFolderStr = rootRecordingFolder;
elseif strcmp(predict_type, "committee_predict")
    relevantFeatures = 1:121;
    relevantFeatures = relevantFeatures';
    global recordingFolders
    % train models on all recording folders
    % note that recording folders must have features data with all EEG
    % channels and with CSP_flag = 0
    CSP_flag = 0;
    wCSPTrainLR = []; wCSPTrainLI = []; wCSPTrainRI = [];
    for ii=1:length(recordingFolders)
        trainRecordingFolder = recordingFolders{ii};    
        pyrunfile("MI5_classifier.py", featuresVariable='AllDataInFeatures', recfolder=trainRecordingFolder, action="train", num_models=num_models);    
    end
    load([trainRecordingFolder '\EEG_chans.mat']);                % load EEG channel names that were used
    recordingFolderStr = strjoin(recordingFolders,";");
end

%% parameters
Fs = 125; 
% Define times
InitWait = 10;                                  % prep time before trials start
bufferPause = 0.2;
trialLength = 5;                                % each trial length in seconds 
cueLength = 2;                                  % time for each cue
readyLength = 1;                                % time "ready" on screen
nextLength = 1;                                 % time "next" on screen

% Define length and classes
numTrials = 10;                             % set number of training trials per class

% Class indices
global numClasses
global idleIndex
global leftIndex
global rightIndex

% Set markers / triggers names
global startRecordings;          
global startTrial;
global Baseline;
global endTrial;
global endRecording;
global EEG_chans_names;

EEG_chans_idx = 1:length(EEG_chans_names);
EEG_chans_idx = EEG_chans_idx(ismember(EEG_chans_names,EEG_chans));



%% Prepare Training Vector
labelsVec = prepareTraining(numTrials,numClasses);    % vector with the conditions for each trial
totalTrials = length(labelsVec);
save(strcat(recordingFolder,'labelsVec.mat'),'labelsVec');

%% Lab Streaming Layer Init
resolveTime = 2; % arbitrary
[outletStream, EEG_Inlet] = LSL_Init(resolveTime);

disp([char(10) 'Open Lab Recorder & check for MarkerStream and EEG stream, start recording, return here and hit any key to continue.']);
pause;                                  % wait for experimenter to press a key

%% Initialize some more variables:
predictionsAll = NaN(1,totalTrials);                % predictions vector
myBuffer = [];                                      % buffer matrix

%% 
pause(bufferPause);                                 % give the system some time to buffer data
myChunk = EEG_Inlet.pull_chunk();                   % get a chunk from the EEG LSL stream to get the buffer going

%% Screen Setup 
monitorPos = get(0,'MonitorPositions'); % monitor position and number of monitors
monitorN = size(monitorPos, 1);
choosenMonitor = 1;                     % which monitor to use                               
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

%% Load images for display - switch to gifs
images{leftIndex} = imread('arrow_left.jpeg', 'jpeg');
images{rightIndex} = imread('arrow_right.jpeg', 'jpeg');
images{idleIndex} = imread('square.jpeg', 'jpeg');

%% Load cue gifs
labelsText{idleIndex} = 'Idle';
labelsText{leftIndex} = 'Left';
labelsText{rightIndex} = 'Right';
[training1Image{leftIndex}, training1Cmap{leftIndex}] = imread([gifs_folder 'inverted hand.gif'],'gif', 'Frames', 'all');
[training1Image{rightIndex}, training1Cmap{rightIndex}] = imread([gifs_folder 'inverted hand other side.gif'],'gif', 'Frames', 'all');
idleFolder = [gifs_folder '\Idle\Funny'];
idleFileList = dir(fullfile(idleFolder, '*.gif'));
finishFolder = 'FinishImages';
finishFileList = dir(fullfile(finishFolder, '*.gif'));
%% open axes for gifs
dimMainLR = 0.4;
heightMainLR = 0.3;
dimMainIdle = 0.5;
heightMainIdle = 0.25;
axTrial(idleIndex) = axes('Position',[(1-dimMainIdle)/2, heightMainIdle, dimMainIdle, dimMainIdle]);
axTrial(leftIndex) = axes('Position',[(1-dimMainLR)/2, heightMainLR, dimMainLR, dimMainLR]);
axTrial(rightIndex) = axes('Position',[(1-dimMainLR)/2, heightMainLR, dimMainLR, dimMainLR]);
for ii = 1:length(axTrial)
    set(axTrial(ii),'color', 'black'); 
    set(axTrial(ii), 'visible', 'off');
end
axFinish = axTrial(idleIndex);
dimClass = 0.4;
axClasses = [];
axClasses(idleIndex) = axes('Position',[(1-dimMainIdle)/2 (heightMainIdle+dimMainIdle-0.1)/2, dimClass, dimClass]);
axClasses(leftIndex) = axes('Position',[0.01 (heightMainLR+dimMainLR)/2-0.01, dimClass, dimClass]);
axClasses(rightIndex) = axes('Position',[(1-dimClass-0.01) (heightMainLR+dimMainLR)/2-0.01, dimClass, dimClass]);
for ii = 1:length(axClasses)
    set(axClasses(ii),'color', 'black'); 
    set(axClasses(ii), 'visible', 'off');
end
axes(hAx)

%%

outletStream.push_sample(startRecordings);      % start of recordings. Later, reject all EEG data prior to this marker
totalTrials = length(labelsVec);
text(0.5,0.5 ,...                               % important for people to prepare
    ['System is calibrating.' newline 'The training session will begin shortly.'], ...
    'HorizontalAlignment', 'Center', 'Color', 'white', 'FontSize', 40);
pause(InitWait)
cla


%% This is the main online script
for trial = 1:totalTrials

    % Display "Next" trial text
    text(0.5,0.5 , 'Next',...
        'HorizontalAlignment', 'Center', 'Color', 'white', 'FontSize', 40);
    % Display trial count
    text(0.5,0.2 , strcat('Trial #',num2str(trial),' Out Of : '...
        ,num2str(totalTrials)),...
        'HorizontalAlignment', 'Center', 'Color', 'white', 'FontSize', 40);
    pause(nextLength);                          % Wait for next trial
    cla                                         % Clear axis
    
    outletStream.push_sample(startTrial);       % trial trigger & counter    
    currentClass = labelsVec(trial);            % What class is it?
    
    % Cue before ready
    text(0.5,0.5 , labelsText{currentClass},...
        'HorizontalAlignment', 'Center', 'Color', 'cyan', 'FontSize', 40);
    
    if any(currentClass == [leftIndex, rightIndex]) 
        play_gifs(training1Image(currentClass), training1Cmap(currentClass),...
            cueLength, [axClasses(currentClass)], hAx)
    else
       pause(cueLength); 
    end
    cla                                 % Clear axis
    
    % Ready
    hReady = text(0.5,0.5 , 'Ready',...
        'HorizontalAlignment', 'Center', 'Color', 'white', 'FontSize', 40);
    outletStream.push_sample(Baseline);         % Baseline trigger
    pause(readyLength);                         % Pause for ready length
    delete(hReady)                              % Clear axis
    
    % start
    % Show gif of the corresponding label of the trial
    outletStream.push_sample(currentClass);     % class label - this indicates the actual start time of the sample
    EEG_Inlet.pull_chunk(); % empty the eeg stream
    if any(currentClass == [leftIndex, rightIndex])
        text(0.5,0.8 , labelsText{currentClass},...
            'HorizontalAlignment', 'Center', 'Color', 'cyan', 'FontSize', 40);
        play_gifs({training1Image{currentClass}},...
            {training1Cmap{currentClass}},...
            trialLength, [axTrial(currentClass)], hAx)
    else
        [training2ImageIdle, training2CmapIdle] = select_random_idle_gif(idleFolder, idleFileList);
        play_gifs({training2ImageIdle},{training2CmapIdle},...
            trialLength, [axTrial(currentClass)], hAx)      
    end
    outletStream.push_sample(endTrial);         % end of trial trigger
        
    myChunk = EEG_Inlet.pull_chunk();           % get data from the inlet from the beginning of the trial

    % and remove a few samples from the beginning and end of the chunk in case they weren't a part of the trial
    myBuffer = myChunk(EEG_chans_idx,3:end-3);
    block = [myBuffer];
    
    cla                                         % Clear axis
    
    % Pre-process the data
    block = MI2_preprocess([], 0, block, Fs, EEG_chans);
    % Extract features from the buffered block:
    EEG_Features = MI4_featureExtraction([], CSP_flag, block, relevantFeatures, wCSPTrainLR, wCSPTrainLI, wCSPTrainRI);
        
    % Predict label
    prediction = pyrunfile("MI5_classifier.py", "prediction", recfolderlist=recordingFolderStr, recfolder=recordingFolderStr, action=predict_type, datapoints=EEG_Features, num_models=num_models);
    % Convert to a Matlab object
    prediction = uint8(double(prediction));
    predictionsAll(trial) = prediction;
    disp(['Predicted label = ' num2str(predictionsAll(trial))])

    % Present result
    displayTrueAndPredicted(currentClass, prediction, labelsText, images);
    pause(2)
    cla
    axes(hAx)
    cla
    pause(0.3)
    hold on

end
outletStream.push_sample(endRecording);          % end of experiment trigger
text(0.5,0.17 , 'Finished!', 'HorizontalAlignment', 'Center', 'Color', 'white', 'FontSize', 40);
[finishImage, finishCmap] = select_random_idle_gif(finishFolder, finishFileList);
play_gifs({finishImage},{finishCmap}, 2*trialLength, [axFinish], hAx);
close
disp('!!!!!!! Stop the LabRecorder recording!');
% save predictions
save(strcat(recordingFolder,'predictionsAll.mat'),'predictionsAll');
accuracy = sum(predictionsAll == labelsVec)/totalTrials;
save(strcat(recordingFolder,'predictionsAllAccuracy.mat'),'accuracy');
disp(['predictionsAll accuracy = ' num2str(accuracy)]);
end

