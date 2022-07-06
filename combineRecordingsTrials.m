function [] = combineRecordingsTrials(recordingFolders, savingFolder)
% Function that combines data from different recording folders
% Run this function after having run MI2 and MI3 on each of the recording
% folders
% Assumption: all recordings use the same EEG_chans (i.e., the same
% EEG channels)
%
% recordingFolders: list of recording folders for combination
%                   E.g., {'recordingFolder1', 'recordingFolder2', 'recordingFolder3'}
% savingFolder: new folder in which to save combination


if ~isfolder(savingFolder)
   mkdir(savingFolder);
end

EEG_chans = cell2mat(struct2cell(load(strcat(recordingFolders{1},'\EEG_chans.mat')))); % load the openBCI channel location

MIData = [];
trainingVec = [];
trials_to_remove = false(0);

for ii=1:length(recordingFolders)
    recordingFolder = recordingFolders{ii};
    MIData_ii = cell2mat(struct2cell(load(strcat(recordingFolder,'\MIData.mat'))));
    if ii > 1
        min_length = min(size(MIData,3), size(MIData_ii,3));
        MIData = cat(1,MIData(:,:,1:min_length), MIData_ii(:,:,1:min_length));
    else
        MIData = MIData_ii;
    end
    trainingVec_ii = cell2mat(struct2cell(load(strcat(recordingFolder,'\trainingVec'))));
    trainingVec = cat(2,trainingVec, trainingVec_ii);
    if isfile(strcat(recordingFolder,'\trials_to_remove.mat'))
        trials_to_remove_ii = cell2mat(struct2cell(load(strcat(recordingFolder,'\trials_to_remove'))));
        trials_to_remove = cat(1,trials_to_remove, trials_to_remove_ii);
    else
        trials_to_remove = cat(1,trials_to_remove, false(1,length(trainingVec)));
    end
end

% save combined data
save(strcat(savingFolder,'\EEG_chans.mat'),'EEG_chans');
save(strcat(savingFolder,'\MIData.mat'),'MIData');
save(strcat(savingFolder,'\trainingVec.mat'),'trainingVec');
save(strcat(savingFolder,'\trials_to_remove.mat'),'trials_to_remove'); 

end

