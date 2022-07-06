function [] = saveProblemTrials(recordingFolder, numTrials, problemTrialsIndices)
% recordingFolder = where the EEG (data & meta-data) are stored.
% numTrials = total number of trials
% problemTrialsIndices = indices of problematic trials (as determined
% during the recording session)

problem_trials = false(1,numTrials);
problem_trials(problemTrialsIndices) = 1;
save(strcat(recordingFolder,'\problem_trials.mat'),'problem_trials');

end

