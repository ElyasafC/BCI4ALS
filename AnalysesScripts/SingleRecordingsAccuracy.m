global_variables
global recordingFolders

%% If didn't run yet, run MI4
CSP_flag = 0;
for jj=1:length(recordingFolders)
    trainRecordingFolder = recordingFolders{jj};
    MI4_featureExtraction(trainRecordingFolder, CSP_flag);
    close all
end

%%
recordings_to_display = [1,2,10,13,14,15];
recordingFoldersToUse = recordingFolders(recordings_to_display);
accuracy = NaN(length(recordingFoldersToUse),2);

CM_flag = 0;
for jj=1:length(recordingFoldersToUse)

    recordingFolder = recordingFoldersToUse{jj};
    if jj == 3
        CM_flag = 1;
    else
        CM_flag = 0;
    end
    means = pyrunfile("MI5_classifier.py", "means", recfolder=recordingFolder, action="test_performance", show_CM=CM_flag);
    means = double(means);
    accuracy(jj,:) = means;
end

figure;
b = bar(accuracy,'FaceColor',"flat");
colors = {[.5 0 .5], [62,150,81]/255};
for k = 1:size(accuracy,2)
    b(k).CData = colors{k};
end
yline(1/3, '--')
legend({'Multiclass', 'Double binary'})
ylabel('Accuracy %')
xlabel('Recording')
box off
