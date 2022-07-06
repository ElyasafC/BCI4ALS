function [] = removeNoisyTrials(recordingFolder, savingFolder)
% Remove trials which have electrodes with very high or very low amplitude
% 
% Run this function after having run MI2, MI3 
% recordingFolder: recording folder with EEG data etc.
% savingFolder: folder in which to save data with removed trials

if ~isfolder(savingFolder)
   mkdir(savingFolder);
end

% load data
load(strcat(recordingFolder,'\EEG_chans.mat'));
load(strcat(recordingFolder,'\MIData.mat')); 
targetLabels = cell2mat(struct2cell(load(strcat(recordingFolder,'\trainingVec'))));
if isfile(strcat(recordingFolder,'\trials_to_remove.mat'))
    trials_to_remove = cell2mat(struct2cell(load(strcat(recordingFolder,'\trials_to_remove'))));
    targetLabels(trials_to_remove) = [];
end

[num_trials, num_electrodes, ~] = size(MIData);
noisy_trials = logical(zeros(num_trials, 1));

threshold = 2; %3;
for ii=1:num_electrodes
%     butterflyPlus(squeeze(MIData(:,ii,:)));   % uncomment if you want to visualize the data
    % remove trials with max or min > threshold*std
    all_max = max(squeeze(MIData(:,ii,:)), [], 2);
    all_min = min(squeeze(MIData(:,ii,:)), [], 2);
    noisy_trials(all_max > threshold*std(all_max)) = 1;
    noisy_trials(abs(all_min) > threshold*std(all_min)) = 1;
end
MIData(noisy_trials,:,:) = [];
targetLabels(noisy_trials) = [];

% save combined data
save(strcat(savingFolder,'\EEG_chans.mat'),'EEG_chans');
save(strcat(savingFolder,'\MIData.mat'),'MIData');
save(strcat(savingFolder,'\targetLabels.mat'),'targetLabels'); 

end

