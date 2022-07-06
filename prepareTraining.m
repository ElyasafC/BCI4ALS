function [trainingVec] = prepareTraining(numTrials,numConditions)
%% return a random vector of 1's, 2's and 3's in the length of numTrials
trainingVecTemp = (1:numConditions);
trainingVec = [];
for i = 1:numTrials
    trainingVec = cat(2, trainingVec,trainingVecTemp(randperm(numConditions)));
    
end