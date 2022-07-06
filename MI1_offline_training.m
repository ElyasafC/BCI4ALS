function [recordingFolder,subID] = MI1_offline_training(rootRecordingFolder)
%% MOTOR IMAGERY Training Scaffolding 
% This code creates a training paradigm with (#) classes on screen for
% (#) numTrials. Before each trial, one of the targets is cued (and remains
% cued for the entire trial).This code assumes EEG is recorded and streamed
% through LSL for later offline preprocessing and model learning.

%% This code is part of the BCI-4-ALS Course written by Asaf Harel
% (harelasa@post.bgu.ac.il) in 2021. You are free to use, change, adapt and
% so on - but please cite properly if published.
% This code was edited by HUJI Team 35 as part of the BCI-4-ALS course.

%% Get global variables
global_variables

%% Make sure you have Lab Streaming Layer installed.
addpath(lsl_lib_path); % lab streaming layer library - update in paths.m to your own computer path
addpath(lsl_bin_path); % lab streaming layer bin - update in paths.m to your own computer path

% Subject and recording parameters:
subID = input('Please enter subject ID/Name: ');    % prompt to enter subject ID or name
if nargin < 1
    rootRecordingFolder = 'C:\Recordings\';                      % define recording folder location
end

% Define recording folder location and create the folder:
recordingFolder = strcat(rootRecordingFolder,'\Sub',num2str(subID),'\');
mkdir(recordingFolder);

% Define times
InitWait = 10;                                  % before trials prep time
trialLength = 5;                                % each trial length in seconds 
cueLength = 2;                                  % time for each cue
readyLength = 1;                                % time "ready" on screen
nextLength = 1;                                 % time "next" on screen

% Define length and classes
numTrials = 20;                                 % set number of training trials per class

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

%% Lab Streaming Layer Init
disp('Loading the Lab Streaming Layer library...');
% Init LSL parameters
lib = lsl_loadlib();                    % load the LSL library
disp('Opening Marker Stream...');
% Define stream parameters
info = lsl_streaminfo(lib,'MarkerStream','Markers',1,0,'cf_string','myuniquesourceid23443');
outletStream = lsl_outlet(info);        % create an outlet stream using the parameters above
disp('Open Lab Recorder & check for MarkerStream and EEG stream, start recording, return here and hit any key to continue.');
pause;                                  % wait for experimenter to press a key

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

%% Prepare Visual Cues - right/left/idle gifs
trainingText{idleIndex} = 'Idle';
trainingText{leftIndex} = 'Left';
trainingText{rightIndex} = 'Right';
[training1Image{leftIndex}, training1Cmap{leftIndex}] = imread([gifs_folder 'inverted hand.gif'],'gif', 'Frames', 'all');
[training1Image{rightIndex}, training1Cmap{rightIndex}] = imread([gifs_folder 'inverted hand other side.gif'],'gif', 'Frames', 'all');
idleFolder = [gifs_folder '\Idle\Funny'];
idleFileList = dir(fullfile(idleFolder, '*.gif'));
finishFolder = 'FinishImages';
finishFileList = dir(fullfile(finishFolder, '*.gif'));
%% Prepare axes for gifs
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

%% Prepare Training Vector
trainingVec = prepareTraining(numTrials,numClasses);    % vector with the conditions for each trial
save(strcat(recordingFolder,'trainingVec.mat'),'trainingVec');

%% Record Training Stage
outletStream.push_sample(startRecordings);      % start of recordings. Later, reject all EEG data prior to this marker
totalTrials = length(trainingVec);
text(0.5,0.5 ,...                               % important for people to prepare
    ['System is calibrating.' newline 'The training session will begin shortly.'], ...
    'HorizontalAlignment', 'Center', 'Color', 'white', 'FontSize', 40);
pause(InitWait)
cla

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
    currentClass = trainingVec(trial);          % What class is it?
    
    % Cue before ready
    text(0.5,0.5 , trainingText{currentClass},...
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
    
    % Show gif of the corresponding label of the trial 
    outletStream.push_sample(currentClass);     % class label - this indicates the actual start time of the sample
    if any(currentClass == [leftIndex, rightIndex])
        text(0.5,0.8 , trainingText{currentClass},...
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

    cla                                         % Clear axis
end

%% End of experiment
outletStream.push_sample(endRecording);          % end of experiment trigger
text(0.5,0.17 , 'Finished!', 'HorizontalAlignment', 'Center', 'Color', 'white', 'FontSize', 40);
[finishImage, finishCmap] = select_random_idle_gif(finishFolder, finishFileList);
play_gifs({finishImage},{finishCmap}, 2*trialLength, [axFinish], hAx);
close
disp('!!!!!!! Stop the LabRecorder recording!');

end

