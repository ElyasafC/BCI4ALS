function MI3_segmentation(recordingFolder)
%% Segment data using markers
% This function segments the continuous data into trials or epochs in a matrix ready for classifier training.

% recordingFolder - where the EEG (data & meta-data) are stored.
% This function assumes MI2_preprocess was already run 

%% This code is part of the BCI-4-ALS Course written by Asaf Harel
% (harelasa@post.bgu.ac.il) in 2021. You are free to use, change, adapt and
% so on - but please cite properly if published.
% This code was adapated by HUJI team 35 as part of the BCI-4-ALS course

%% Parameters and previous variables:
Fs = 125;               % openBCI sample rate
load(strcat(recordingFolder,'\cleaned_sub.mat'));               % load the clean EEG data in .mat format
if isfile(strcat(recordingFolder,'\trainingVec.mat'))
    load(strcat(recordingFolder,'\trainingVec.mat'));           % load the training vector (which target at which trial)
else
    trainingVec = load(strcat(recordingFolder,'\labelsVec.mat')).labelsVec; % load the labels vector (vector saved from online feedback)
end
load(strcat(recordingFolder,'\EEG_chans.mat'));                 % load the EEG channel locations
numChans = length(EEG_chans);                                   % how many chans do we have?
load(strcat(recordingFolder,'\EEG_events.mat'));                % load the EEG event markers


%% variables

global_variables

global startTrial Baseline endTrial
global idleIndex leftIndex rightIndex

%% Extract trials through the events
trials1 = length(trainingVec);                                  % derive number of trials from training label vector
events = struct('type', {EEG_event(1:end).type});
left = []; right = []; idle = [];
for i = 1:length(events)
    if str2num(EEG_event(i).type) == startTrial                 % find trial start marker
        marker1Index(i) = 1;                                    % index markers
    else
        marker1Index(i) = 0;
    end
    
    if str2num(EEG_event(i).type) == Baseline                   % find trial baseline start marker
        startBaseline(i) = 1;                                   % index markers
    else
        startBaseline(i) = 0;
    end
    
    if any(str2num(EEG_event(i).type) == [idleIndex, leftIndex, rightIndex])   % find start marker of actual class
        startClass(i) = 1;
    else
        startClass(i) = 0;
    end
    
    if str2num(EEG_event(i).type) == endTrial                          % find end marker of actual class
        endClass(i) = 1;
    else
        endClass(i) = 0;
    end
    
    if str2num(EEG_event(i).type) == leftIndex
        left(end+1) = EEG_event(i).latency;
    elseif str2num(EEG_event(i).type) == rightIndex
        right(end+1) = EEG_event(i).latency;
    elseif str2num(EEG_event(i).type) == idleIndex
        idle(end+1) =  EEG_event(i).latency;
    end
end
all_trials = [left, right, idle];
all_trials_sorted = sort(all_trials);
diff(all_trials_sorted)/Fs
figure; histogram(diff(all_trials_sorted)/Fs); 
title('Time between each 2 trials'); xlabel('Time (s)')
trials = min(sum(startClass),sum(endClass));                        % derive number of trials from start markers

% Check for consistancy across events & trials
if trials ~= trials1
    disp('!!!! Some form of mis-match between number of recorded and planned trials.')
%     return
end

startClassIndex = find(startClass);
endClassIndex = find(endClass);

%% Main data segmentation process:
MIData = NaN([trials, size(EEG_data, 1), size(EEG_data, 2)]);
for trial = 1:trials
    [MIData] = sortElectrodes(MIData,EEG_data,EEG_event,startClassIndex(trial),endClassIndex(trial),numChans,trial);
end

% remove last columns if trials don't all have this data (note that all
% channels should have nans if trial doesn't have the data)
trial_lengths_all = sum(~isnan(squeeze(MIData(:,1,:))),2);
cut_off = 580; % this is appropriate for trialLength = 5, CHANGE THIS IF YOU CHANGE TRIAL LENGTH (should be a little less than trialLength*Fs)
trials_to_remove = (trial_lengths_all <= cut_off);
% remove problem trials (as determined during the recording session and saved in folder using "saveProblemTrials.m")
if isfile(strcat(recordingFolder,'\problem_trials.mat'))
   problem_trials = cell2mat(struct2cell(load(strcat(recordingFolder,'\problem_trials')))); 
   if trials == length(problem_trials)
        trials_to_remove(problem_trials) = true;   
    end
end
MIData(trials_to_remove,:,:) = [];
if isfile(strcat(recordingFolder,'\problem_trials.mat')) && (trials ~= length(problem_trials))
    trials_to_remove(problem_trials) = true;   
end
idx_nan_first = find(sum(isnan(squeeze(MIData(:,1,:))),1) > 0, 1, 'first');
MIData(:,:,idx_nan_first:end) = [];


save(strcat(recordingFolder,'\MIData.mat'),'MIData');                      % save sorted data
save(strcat(recordingFolder,'\trials_to_remove.mat'),'trials_to_remove');  % save trials that were removed

end
