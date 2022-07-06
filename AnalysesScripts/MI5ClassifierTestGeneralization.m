global_variables
global recordingFolders


%% Test how classifier trained on each recording generalizes on other recordings

%% If didn't run yet, run MI4 with CSP_flag = 0 or 1
% Note that CSP_flag = 1 will add noise to generalization
CSP_flag = 0;
for jj=1:length(recordingFolders)
    trainRecordingFolder = recordingFolders{jj};
    MI4_featureExtraction(trainRecordingFolder, CSP_flag);
    close all
end
%%
num_models = "1"; % 1 - multiclass classifier, 2 - double binary classifier

accuracy = NaN(length(recordingFolders),length(recordingFolders));

for jj=1:length(recordingFolders)

    trainRecordingFolder = recordingFolders{jj};

    relevantFeatures = load([trainRecordingFolder '\AllDataTopFeaturesIdx.mat']).AllDataTopFeaturesIdx;
    CSP_flag = load([trainRecordingFolder '\CSP_flag.mat']).CSP_flag;
    if CSP_flag
        EEG_chans = load([trainRecordingFolder '\EEG_chans.mat']).EEG_chans;
        wCSPTrainLR = load([trainRecordingFolder '\wCSPTrainLR.mat']).wCSPTrainLR;                   % load weights of CSP left vs right
        wCSPTrainLI = load([trainRecordingFolder '\wCSPTrainLI.mat']).wCSPTrainLI;                   % load weights of CSP left vs idle
        wCSPTrainRI = load([trainRecordingFolder '\wCSPTrainRI.mat']).wCSPTrainRI;                   % load weights of CSP right vs idle
    end

    pyrunfile("MI5_classifier.py", featuresVariable='AllDataTopFeatures', recfolder=trainRecordingFolder, action="train", num_models=num_models);
    
    for ii=1:length(recordingFolders)
        testRecordingFolder = recordingFolders{ii};
        if CSP_flag
            % extract features using trained CSP weights
            MIData = load([testRecordingFolder '\MIData.mat']).MIData;
            AllDataRelevantFeatures = MI4_featureExtraction([], CSP_flag, MIData, relevantFeatures, wCSPTrainLR, wCSPTrainLI, wCSPTrainRI);
        else
            % extract saved features
            AllDataInFeatures = load([testRecordingFolder '\AllDataInFeatures.mat']).AllDataInFeatures;
            AllDataRelevantFeatures = AllDataInFeatures(:,relevantFeatures);
        end
        if ~strcmp(testRecordingFolder, trainRecordingFolder)
            AllDataLabels = load([testRecordingFolder '\AllDataLabels.mat']).AllDataLabels;
            predictions = pyrunfile("MI5_classifier.py", "prediction", recfolder=trainRecordingFolder, action="predict", datapoints=AllDataRelevantFeatures, num_models=num_models);
            predictions = double(predictions);
            accuracy(jj,ii) = sum(predictions == AllDataLabels)/length(AllDataLabels);
        else
            means = pyrunfile("MI5_classifier.py", "means", recfolder=testRecordingFolder, action="test_performance", show_CM=0);
            means = double(means);
            accuracy(jj,ii) = means(2);
        end
    end
end

figure;
hm = heatmap(accuracy);
origState = warning('query', 'MATLAB:structOnObject');
cleanup = onCleanup(@()warning(origState));
warning('off','MATLAB:structOnObject')
S = struct(hm);
ax = S.Axes;
clear('cleanup')
hm.GridVisible = 'off';
daySeparators = [1,4,7,10,13];  % indices of new recording days. Change this according to recording folders list.
xline(ax, daySeparators+.5, 'k-');
yline(ax, daySeparators+.5, 'k-');
title('Recordings Generalization')


%% Test how committee generalizes

%% If didn't run yet, run MI4 with CSP_flag = 0
% NOTE: CSP needs to be 0 since we're using multiple models trained on
% different datasets (i.e., no common CSP weights)
CSP_flag = 0;
for jj=1:length(recordingFolders)
    trainRecordingFolder = recordingFolders{jj};
    MI4_featureExtraction(trainRecordingFolder, CSP_flag);
    close all
end
%%
num_models = "2"; % 1 - multiclass classifier, 2 - double binary classifier

reps = 20;
accuracyCommittee = NaN(reps,length(recordingFolders));
for jj=1:reps
    for ii=1:length(recordingFolders)
        trainRecordingFolder = recordingFolders{ii};    
        pyrunfile("MI5_classifier.py", featuresVariable='AllDataInFeatures', recfolder=trainRecordingFolder, action="train", num_models=num_models);    
    end
    
    allRecordingsIndices = 1:length(recordingFolders);
    
    for ii=1:length(recordingFolders)
        recordingsIndices = allRecordingsIndices;
        recordingsIndices(ii) = [];  % remove test recording from train committee
        recordingFoldersCopy = recordingFolders(recordingsIndices);
        recordingFoldersCopy = strjoin(recordingFoldersCopy,";");
        testRecordingFolder = recordingFolders{ii};
        AllDataInFeatures = load([testRecordingFolder '\AllDataInFeatures.mat']).AllDataInFeatures;
        AllDataLabels = load([testRecordingFolder '\AllDataLabels.mat']).AllDataLabels;
        predictions = pyrunfile("MI5_classifier.py", "prediction", recfolderlist=recordingFoldersCopy, action="committee_predict", datapoints=AllDataInFeatures, num_models=num_models);
        predictions = double(predictions);
        accuracyCommittee(jj,ii) = sum(predictions == AllDataLabels)/length(AllDataLabels); 
    end
end
accuracyCommitteeMean = mean(accuracyCommittee, 1); 
figure;
scatter(1:length(recordingFolders), accuracyCommitteeMean, 'filled')
yline(1/3, '--')
xlabel('Recording')
ylabel('Accuracy')
title('Committee classification')