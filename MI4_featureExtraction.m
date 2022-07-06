function [AllDataInFeatures] = MI4_featureExtraction(recordingFolder, CSP_flag, MIData, relevantFeatures, wCSPTrainLR, wCSPTrainLI, wCSPTrainRI)

%% This function extracts features for the machine learning process.
% Computes the best common spatial patterns from all available
% labeled training trials. The next part extracts all learned features.
% This includes a non-exhaustive list of possible features (commented below).
% At the bottom there is a simple feature importance test that chooses the
% best features and saves them for model training.

% For offline feature extraction - this function assumes MI3 has already been run 

% For OFFLINE preprocessing:
% recordingFolder - where the EEG (data & meta-data) are stored.

% CSP_flag: 1 or 0, if not entered, CSP_flag will be set to 0 (i.e., CSP
% (Common Spatial Patterns) features won't be included)

% ADAPTATION FOR ONLINE LEARNING:
% Instead of "recordingFolder", need the following parameters:
% (1) MIData - clean EEG data segmented into trials (could be one trial) 
% (2) relevantFeatures - selected feature indices to compute
% (3) wCSPTrainLR, wCSPTrainLI, wCSPTrainRI - CSP weights, required if
% CSP_flag is 1

%% This code is part of the BCI-4-ALS Course written by Asaf Harel
% (harelasa@post.bgu.ac.il) in 2021. You are free to use, change, adapt and
% so on - but please cite properly if published.
% This code was adapated by HUJI team 35 as part of the BCI-4-ALS course

%%
if nargin < 2 || isempty(CSP_flag)
    % default
    CSP_flag = 0;
end

%% Load previous variables:
if ~isempty(recordingFolder)
    % For offline processing
    offline = true;
    load(strcat(recordingFolder,'\EEG_chans.mat'));                  % load the openBCI channel location
    load(strcat(recordingFolder,'\MIData.mat'));                     % load the EEG data
    if isfile(strcat(recordingFolder,'\targetLabels.mat'))
        targetLabels = cell2mat(struct2cell(load(strcat(recordingFolder,'\targetLabels'))));
    else
        if isfile(strcat(recordingFolder,'\trainingVec.mat'))
            targetLabels = cell2mat(struct2cell(load(strcat(recordingFolder,'\trainingVec'))));
        else
            targetLabels = load(strcat(recordingFolder,'\labelsVec.mat')).labelsVec;               % load the training vector (which target at which trial)
        end
        if isfile(strcat(recordingFolder,'\trials_to_remove.mat'))
            trials_to_remove = cell2mat(struct2cell(load(strcat(recordingFolder,'\trials_to_remove'))));
            targetLabels(trials_to_remove) = [];
        end
    end
else
    % online learning
    offline = false;
    numClasses = 1;
    if nargin < 3 || isempty(MIData)
       disp('Please enter clean EEG data as MIData')
       return
    elseif nargin < 4 || isempty(relevantFeatures) 
       disp('Please enter indices of relevant features')
       return
    elseif CSP_flag
        if any(ismember(1:3, relevantFeatures)) && (nargin < 5 || isempty(wCSPTrainLR))
           disp('Please enter weights of CSP left vs right')
           return
        elseif any(ismember(4:6, relevantFeatures)) && (nargin < 6 || isempty(wCSPTrainLI))
           disp('Please enter weights of CSP left vs right')
           return
        elseif any(ismember(7:9, relevantFeatures)) && (nargin < 7 || isempty(wCSPTrainRI))
           disp('Please enter weights of CSP left vs right')
           return
        end
    end
    if length(size(MIData)) == 2
        trials = 1;
        MIData_3d(1,:,:) = MIData;
        MIData = MIData_3d;
    else
        trials = size(MIData, 1);
    end
end

global_variables
global idleIndex
global leftIndex
global rightIndex
labelsOrder = sort([idleIndex, leftIndex, rightIndex]);
labels_ = {'idle', 'left', 'right'};
labels = labels_(labelsOrder);
Fs = 125;                                                           % openBCI Cyton+Daisy by Bluetooth sample rate
numChans = size(MIData,2);                                          % get number of channels from main data variable
if offline
    Features2Select = 10;                                           % number of featuers for feature selection
    num4test = 5;                                                   % define how many test trials after feature extraction
    numClasses = length(unique(targetLabels));                      % set number of possible targets (classes)
    trials = size(MIData,1);                                        % get number of trials from main data variable
    [R, C] = size(EEG_chans);                                       % get EEG_chans (char matrix) size - rows and columns
    chanLocs = reshape(EEG_chans',[1, R*C]);                        % reshape into a vector in the correct order
end
%% Visual Feature Selection: Power Spectrum

motorDataChan = {};
welch = {};
idxTarget = {};
freq.low = 0.5;                             % INSERT the lowest freq 
freq.high = 40;                             % INSERT the highst freq 
freq.Jump = 1;                              % SET the freq resolution
f = freq.low:freq.Jump:freq.high;           % frequency vector
window = 40;                                % INSERT sample size window for pwelch
noverlap = 20;                              % INSERT number of sample overlaps for pwelch
vizChans = [1,2];                           % INSERT which 2 channels you want to compare
if offline
    % create power spectrum figure:
    f1 = figure('name','PSD','NumberTitle','off');
    sgtitle(['FC Electrodes' char(10) 'Power Spectrum']);
    f2 = figure('name','PSD','NumberTitle','off');
    sgtitle(['CP Electrodes' char(10) 'Power Spectrum']);
    f3 = figure('name','PSD','NumberTitle','off');
    sgtitle(['Power Spectrum']);
    f4 = figure('name','PSD','NumberTitle','off');
    sgtitle(['O & Cz Electrodes' char(10) 'Power Spectrum']);
end
% compute power Spectrum per electrode in each class
psd = nan(numChans,numClasses,2,1000); % init psd matrix
allSpect = [];
if offline
    f1_idx = 1; f2_idx = 1; f3_idx = 1; f4_idx = 1;
end
for chan = 1:numChans
    motorDataChan{chan} = squeeze(MIData(:,chan,:))';                   % convert the data to a 2D matrix fillers by channel
    nfft = 2^nextpow2(size(motorDataChan{chan},1));                     % take the next power of 2 length of the specific trial length
    tempWelch = pwelch(motorDataChan{chan},window, noverlap, f, Fs);  % calculate the pwelch for each electrode
    if size(tempWelch,1) == 1
        tempWelch = tempWelch';
    end
    welch{chan} = tempWelch;
    if offline
        if contains(EEG_chans(chan,:), 'FC')
            figure(f1);
            subplot(2,2,f1_idx);
            f1_idx = f1_idx + 1;
        elseif contains(EEG_chans(chan,:), 'CP')
            figure(f2);
            subplot(2,2,f2_idx);
            f2_idx = f2_idx + 1;
        elseif strcmp(EEG_chans(chan,:), 'C03') || strcmp(EEG_chans(chan,:), 'C04')
            figure(f3);
            subplot(1,2,f3_idx);
            f3_idx = f3_idx + 1;
        else
            figure(f4);
            subplot(1,3,f4_idx); 
            f4_idx = f4_idx +1; 
        end
        for class = 1:numClasses
            idxTarget{class} = find(targetLabels == class);                 % find the target index
            plot(f, log10(mean(welch{chan}(:,idxTarget{class}), 2)));       % ploting the mean power spectrum in dB by each channel & class
            hold on
            for trial = 1:length(idxTarget{class})                          % run over all concurrent class trials
                [s,spectFreq,t,psd] = spectrogram(motorDataChan{chan}(:,idxTarget{class}(trial)),window,noverlap,nfft,Fs);  % compute spectrogram on specific channel
                multiPSD(trial,:,:) = psd;
            end

            % compute mean spectrogram over all trials with same target
            totalSpect(chan,class,:,:) = squeeze(mean(multiPSD,1));
            allSpect(idxTarget{class},chan,:,:) = multiPSD;
            clear multiPSD psd
        end
        title([EEG_chans(chan,:)])
        legend(labels)
        xlabel('Freq. (Hz)')
    end
end
if offline
   figure(f1); linkaxes
   figure(f2); linkaxes
   figure(f3); linkaxes
   figure(f4); linkaxes   
end

if offline
    % plot each trial for chosen channel
    chan = 1;   
    blue = [0, 0, 1];
    red = [1, 0, 0];
    yellow = [1, 1, 0];
    black = [0,0,0];
    figure; hold on;
    color_gradient = @(c1, c2, N) [linspace(c1(1),c2(1),N)', linspace(c1(2),c2(2),N)', linspace(c1(3),c2(3),N)'];
    colororder(color_gradient(blue,black,length(idxTarget{1})));
    plot(f, log10(welch{chan}(:,idxTarget{1}))); title([EEG_chans(chan,:) ': ' labels{1} ' trials'])
    figure;
    colororder(color_gradient(red,black,length(idxTarget{2})));
    plot(f, log10(welch{chan}(:,idxTarget{2})));  title([EEG_chans(chan,:) ': ' labels{2} ' trials'])
    figure;
    colororder(color_gradient(yellow,black,length(idxTarget{3})));
    plot(f, log10(welch{chan}(:,idxTarget{3})));  title([EEG_chans(chan,:) ': ' labels{3} ' trials'])
end


if offline
    % To look at single trials
    trial = 1; 
    butterflyPlus(squeeze(MIData(trial,:,:)))
    title(['Trial ' num2str(trial) ' (all electrodes)'])
    xlabel('points in time');
    ylabel('amplitude');
end

if offline
    idleIdx = find(targetLabels == idleIndex);                  % find idle trials
    leftIdx = find(targetLabels == leftIndex);                  % find left trials
    rightIdx = find(targetLabels == rightIndex);                 % find right trials
end

%% Common Spatial Patterns
% create a spatial filter using available EEG & labels
% we will "train" a mixing matrix (wCSPTrainLR) on 80% of the trials and another
% mixing matrix (wViz) just for the visualization trial (vizTrial). This
% serves to show an understandable demonstration of the process.

if CSP_flag
    cspChannels = 1:numChans;
end
if offline && CSP_flag
    % Begin by splitting into two classes:
    leftClass = MIData(targetLabels == leftIndex,:,:);
    rightClass = MIData(targetLabels == rightIndex,:,:);
    idleClass = MIData(targetLabels == idleIndex,:,:);

    % Aggregate all trials into one matrix
    overallLeft = [];
    overallRight = [];
    overallIdle = [];
    rightIndices = rightIdx(randperm(length(rightIdx)));% randomize right indexs
    leftIndices  = leftIdx(randperm(length(leftIdx)));   % randomize left indexs
    idleIndices  = idleIdx(randperm(length(idleIdx)));   % randomize idle indexs
    minTrials = min([length(leftIndices), length(rightIndices)]);
    percentIdx = floor(0.8*minTrials);                  % this is the 80% part...
    for trial=1:percentIdx
        overallLeft = [overallLeft squeeze(MIData(leftIndices(trial),:,:))];
        overallRight = [overallRight squeeze(MIData(rightIndices(trial),:,:))];
        overallIdle = [overallIdle squeeze(MIData(idleIndices(trial),:,:))];
    end
    vizTrial = 3;       % cherry-picked!
    % visualize the CSP data:
    figure;
    subplot(1,2,1)      % show a single trial before CSP seperation
    scatter3(squeeze(leftClass(vizTrial,1,:)),squeeze(leftClass(vizTrial,2,:)),squeeze(leftClass(vizTrial,3,:)),'b'); hold on
    scatter3(squeeze(rightClass(vizTrial,1,:)),squeeze(rightClass(vizTrial,2,:)),squeeze(rightClass(vizTrial,3,:)),'g');
    title('Before CSP')
    legend('Left','Right')
    xlabel('channel 1')
    ylabel('channel 2')
    zlabel('channel 3')
    % find mixing matrix (wAll) for all trials:
    [wCSPTrainLR, lambdaLR, A] = csp(overallLeft(cspChannels,:), overallRight(cspChannels,:));
    [wCSPTrainLI, lambdaLI, A] = csp(overallLeft(cspChannels,:), overallIdle(cspChannels,:));
    [wCSPTrainRI, lambdaRI, A] = csp(overallRight(cspChannels,:), overallIdle(cspChannels,:));
    % save weights
    save(strcat(recordingFolder,'\wCSPTrainLR.mat'),'wCSPTrainLR');
    save(strcat(recordingFolder,'\wCSPTrainLI.mat'),'wCSPTrainLI');
    save(strcat(recordingFolder,'\wCSPTrainRI.mat'),'wCSPTrainRI');

    % find mixing matrix (wViz) just for visualization trial:  (for left vs right)
    [wViz, lambdaViz, Aviz] = csp(squeeze(leftClass(vizTrial,cspChannels,:)), squeeze(rightClass(vizTrial,cspChannels,:)));
    % apply mixing matrix on available data (for visualization)
    leftClassCSP = (wViz'*squeeze(leftClass(vizTrial,cspChannels,:)));
    rightClassCSP = (wViz'*squeeze(rightClass(vizTrial,cspChannels,:)));

    subplot(1,2,2)      % show a single trial after CSP seperation (for left vs right)
    scatter3(squeeze(leftClassCSP(1,:)),squeeze(leftClassCSP(2,:)),squeeze(leftClassCSP(length(lambdaViz),:)),'b'); hold on
    scatter3(squeeze(rightClassCSP(1,:)),squeeze(rightClassCSP(2,:)),squeeze(rightClassCSP(length(lambdaViz),:)),'g');
    title('After CSP')
    legend('Left','Right')
    xlabel('CSP dimension 1')
    ylabel('CSP dimension 2')
    zlabel('CSP dimension 3')

end
%% Spectral frequencies and times for bandpower features:
% Choose frequency bands that will be used as features
bands{1} = [5, 15];
bands{2} = [15, 30];
numSpectralFeatures = length(bands);                        % how many features exist overall 
%% Extract features 
% Note - in online setting, only relevantFeatures will be computed
if CSP_flag
    num_CSP_features = 9;
else
    num_CSP_features = 0;
end
numAdditionalFeatures = 9;
MIFeaturesLabel = NaN(trials,numChans,numSpectralFeatures+numAdditionalFeatures); % init features + labels matrix
MIFeaturesName = cell(trials,numChans,numSpectralFeatures+numAdditionalFeatures); % names of features
CSPFeatures = NaN(trials, num_CSP_features);
for trial = 1:trials                                % run over all the trials
    featureIndx = 1;
    
    if CSP_flag
        % CSP: left vs right 
        % (using W computed above for all channels at once)
        if offline || any(ismember(1:3, relevantFeatures))
            temp = var((wCSPTrainLR'*squeeze(MIData(trial,cspChannels,:)))');    % apply the CSP filter on the current trial EEG data
            CSPFeatures(trial,1:3) = temp([1,2,length(cspChannels)]);            % add the variance from the first 3 eigenvalues
            clear temp                                                           % clear the variable to free it for the next loop
        end
        % CSP: left vs idle
        if offline || any(ismember(4:6, relevantFeatures))
            temp = var((wCSPTrainLI'*squeeze(MIData(trial,cspChannels,:)))');          % apply the CSP filter on the current trial EEG data
            CSPFeatures(trial,4:6) = temp([1,length(cspChannels)-1,length(cspChannels)]);     % add the variance from the first 3 eigenvalues
            clear temp                                                              % clear the variable to free it for the next loop
        end
        % CSP: right vs idle
        if offline || any(ismember(7:9, relevantFeatures))
            temp = var((wCSPTrainRI'*squeeze(MIData(trial,cspChannels,:)))');      % apply the CSP filter on the current trial EEG data
            CSPFeatures(trial,7:9) = temp([1,2,length(cspChannels)]);                % add the variance from the first 3 eigenvalues
            clear temp                                                          % clear the variable to free it for the next loop
        end
        featureIndx = featureIndx + num_CSP_features;
    end
    for channel = 1:numChans                        % run over all the electrodes (channels)
        n = 1;                                      % start a new feature index
        for feature = 1:numSpectralFeatures                 % run over all spectral band power features from the section above
            % Extract features: bandpower +-1 Hz around each target frequency
            if offline || any(ismember(featureIndx, relevantFeatures))
                MIFeaturesLabel(trial,channel,n) = bandpower(squeeze(MIData(trial,channel,:)),Fs,bands{feature});
                if offline
                    MIFeaturesName{trial,channel,n} = [EEG_chans(channel,:) ': bandpower ' num2str(bands{feature}(1)) '-' num2str(bands{feature}(2)) ' Hz '];
                end
            end
            n = n+1;  
            featureIndx = featureIndx + 1;
        end
        if offline
            disp(strcat('Extracted Powerbands from electrode:',EEG_chans(channel,:)))            
        end
        
        % NOVEL Features - an explanation for each can be found in the class presentation folder
        
        % Normalize the Pwelch matrix
        if offline || any(ismember(featureIndx:featureIndx+numAdditionalFeatures-1, relevantFeatures))
            pfTot = sum(welch{channel}(:,trial));               % Total power for each trial
            normlizedMatrix = welch{channel}(:,trial)./pfTot;   % Normalize the Pwelch matrix by dividing the matrix in its sum for each trial
        end
        if offline
            disp(strcat('Extracted Normalized Pwelch Matrix from electrode:',EEG_chans(channel,:)))            
        end
        
        % Root Total Power
        if offline || any(ismember(featureIndx, relevantFeatures))
            MIFeaturesLabel(trial,channel,n) = sqrt(pfTot);     % Square-root of the total power
        end
        if offline
            MIFeaturesName{trial,channel,n} = [EEG_chans(channel,:) ': square-root of total power'];
            disp(strcat('Extracted Root Total Power from electrode:',EEG_chans(channel,:)))            
        end
        n = n + 1;
        featureIndx = featureIndx + 1;
        
        % Spectral Moment
        if offline || any(ismember(featureIndx, relevantFeatures))
            MIFeaturesLabel(trial,channel,n) = sum(normlizedMatrix.*f'); % Calculate the spectral moment
        end
        if offline
            MIFeaturesName{trial,channel,n} = [EEG_chans(channel,:) ': spectral moment'];
            disp(strcat('Extracted Normalized Pwelch Matrix from electrode:',EEG_chans(channel,:)))           
        end
        n = n + 1;
        featureIndx = featureIndx + 1;
        
        % Spectral Edge
        if offline || any(ismember(featureIndx, relevantFeatures))
            probfunc = cumsum(normlizedMatrix);                 % Create matrix of cumulative sum
            % frequency that 90% of the power resides below it and 10% of the power resides above it
            valuesBelow = @(z)find(probfunc(:,z)<=0.9);         % Create local function
            % apply function to each element of normlizedMatrix
            fun4Values = arrayfun(valuesBelow, 1:size(normlizedMatrix',1), 'un',0);
            lengthfunc = @(y)length(fun4Values{y})+1;           % Create local function for length
            % apply function to each element of normlizedMatrix
            fun4length = cell2mat(arrayfun(lengthfunc, 1:size(normlizedMatrix',1), 'un',0));
            MIFeaturesLabel(trial,channel,n) = f(fun4length);   % Insert it to the featurs matrix
        end
        if offline
            MIFeaturesName{trial,channel,n} = [EEG_chans(channel,:) ': spectral edge'];
            disp(strcat('Extracted Spectral Edge from electrode:',EEG_chans(channel,:)))            
        end
        n = n + 1;
        featureIndx = featureIndx + 1;
            
        % Spectral Entropy
        if offline || any(ismember(featureIndx, relevantFeatures))
            MIFeaturesLabel(trial,channel,n) = -sum(normlizedMatrix.*log2(normlizedMatrix)); % calculate the spectral entropy
        end
        if offline
            MIFeaturesName{trial,channel,n} = [EEG_chans(channel,:) ': spectral entropy'];
            disp(strcat('Extracted Spectral Entropy from electrode:',EEG_chans(channel,:)))            
        end
        n = n + 1;
        featureIndx = featureIndx + 1;
            
        % Slope
        transposeMat = (welch{channel}(:,trial)');          % transpose matrix
        % create local function for computing the polyfit on the transposed matrix and the frequency vector
        FitFH = @(k)polyfit(log(f(1,:)),log(transposeMat(k,:)),1);
        % convert the cell that gets from the local func into matrix, perform the
        % function on transposeMat, the slope is in each odd value in the matrix
        % Apply function to each element of tansposeMat
        pFitLiner = cell2mat(arrayfun(FitFH, 1:size(transposeMat,1), 'un',0));
        if offline || any(ismember(featureIndx, relevantFeatures))
            MIFeaturesLabel(trial,channel,n)=pFitLiner(1:2 :length(pFitLiner));
        end
        if offline
            MIFeaturesName{trial,channel,n} = [EEG_chans(channel,:) ': slope'];
            disp(strcat('Extracted Slope from electrode:',EEG_chans(channel,:)))            
        end
        n = n + 1;
        featureIndx = featureIndx + 1;
        
        % Intercept
        % the slope is in each double value in the matrix
        if offline || any(ismember(featureIndx, relevantFeatures))
            MIFeaturesLabel(trial,channel,n)=pFitLiner(2:2:length(pFitLiner));
        end
        if offline
            MIFeaturesName{trial,channel,n} = [EEG_chans(channel,:) ': intercept'];
            disp(strcat('Extracted Intercept from electrode:',EEG_chans(channel,:)))            
        end
        n = n + 1;
        featureIndx = featureIndx + 1;
        
        % Mean Frequency
        % returns the mean frequency of a power spectral density (PSD) estimate, pxx.
        % The frequencies, f, correspond to the estimates in pxx.
        if offline || any(ismember(featureIndx, relevantFeatures))  
            MIFeaturesLabel(trial,channel,n) = meanfreq(normlizedMatrix,f);
        end
        if offline
            MIFeaturesName{trial,channel,n} = [EEG_chans(channel,:) ': mean frequency'];
            disp(strcat('Extracted Mean Frequency from electrode:',EEG_chans(channel,:)))            
        end
        n = n + 1;
        featureIndx = featureIndx + 1;
        
        % Occupied bandwidth
        % returns the 99% occupied bandwidth of the power spectral density (PSD) estimate, pxx.
        % The frequencies, f, correspond to the estimates in pxx.
        if offline || any(ismember(featureIndx, relevantFeatures))
            MIFeaturesLabel(trial,channel,n) = obw(normlizedMatrix,f);
        end
        if offline
            MIFeaturesName{trial,channel,n} = [EEG_chans(channel,:) ': occupied bandwidth'];
            disp(strcat('Extracted Occupied bandwidth from electrode:',EEG_chans(channel,:)))            
        end
        n = n + 1;
        featureIndx = featureIndx + 1;
        
        % Power bandwidth
        if offline || any(ismember(featureIndx, relevantFeatures))
            MIFeaturesLabel(trial,channel,n) = powerbw(normlizedMatrix,Fs);
        end 
        if offline
            MIFeaturesName{trial,channel,n} = [EEG_chans(channel,:) ': power bandwidth'];
            disp(strcat('Extracted Power bandwidth from electrode:',EEG_chans(channel,:)))            
        end
        n = n + 1;
        featureIndx = featureIndx + 1;
    end
end


MIFeaturesLabel = zscore(MIFeaturesLabel);

% Reshape into 2-D matrix
numFeaturesChannels = size(MIFeaturesLabel,2)*size(MIFeaturesLabel,3);
MIFeatures = NaN(trials,numFeaturesChannels);
for ii=1:size(MIFeaturesLabel,1)
    for jj=1:size(MIFeaturesLabel,2)
        MIFeatures(ii,(jj-1)*size(MIFeaturesLabel,3)+1:jj*size(MIFeaturesLabel,3)) = MIFeaturesLabel(ii,jj,:);
    end
end
if offline
    % reshape names matrix
    MIFeaturesName2D = reshape(MIFeaturesName, trials, []);
end
if CSP_flag
    MIFeatures = [CSPFeatures MIFeatures];              % add the CSP features to the overall matrix
end
if offline
    if CSP_flag
        MIFeaturesName2D = [repmat({'CSP LR1'},1,size(MIFeaturesName2D,1))'...
                            repmat({'CSP LR2'},1,size(MIFeaturesName2D,1))'...
                            repmat({'CSP LR11'},1,size(MIFeaturesName2D,1))'...
                            repmat({'CSP LI1'},1,size(MIFeaturesName2D,1))'...
                            repmat({'CSP LI10'},1,size(MIFeaturesName2D,1))'...
                            repmat({'CSP LI11'},1,size(MIFeaturesName2D,1))'...
                            repmat({'CSP RI1'},1,size(MIFeaturesName2D,1))'...
                            repmat({'CSP RI2'},1,size(MIFeaturesName2D,1))'...
                            repmat({'CSP RI11'},1,size(MIFeaturesName2D,1))'...                                                                           
                            MIFeaturesName2D]; 
    end
end
                
AllDataInFeatures = MIFeatures;
if offline
    save(strcat(recordingFolder,'\AllDataInFeatures.mat'),'AllDataInFeatures');
    AllDataLabels = targetLabels;
    save(strcat(recordingFolder,'\AllDataLabels.mat'),'AllDataLabels');
else
    IrrelevantFeatuers = 1:size(AllDataInFeatures,2);
    IrrelevantFeatuers(relevantFeatures) = []; 
    AllDataInFeatures(:,IrrelevantFeatuers) = []; 
end

if offline
    %% Split to training and test sets
    min_trials = min([length(idleIdx), length(leftIdx), length(rightIdx)]);
    testIdx = randperm(min_trials,num4test);                            % picking test index randomly
    testIdx = [idleIdx(testIdx) leftIdx(testIdx) rightIdx(testIdx)];    % taking the test index from each class
    testIdx = sort(testIdx);                                            % sort the trials

    % split test data
    FeaturesTest = MIFeatures(testIdx,:,:);     % taking the test trials features from each class
    LabelTest = targetLabels(testIdx);          % taking the test trials labels from each class

    % split train data
    FeaturesTrain = MIFeatures;
    FeaturesTrain (testIdx ,:,:) = [];          % delete the test trials from the features matrix, and keep only the train trials
    LabelTrain = targetLabels;
    LabelTrain(testIdx) = [];                   % delete the test trials from the labels matrix, and keep only the train labels


    %% Feature Selection (using neighborhood component analysis)
    class = fscnca(FeaturesTrain,LabelTrain);   % feature selection
    % sorting the weights in desending order and keeping the indexs
    [~,selected] = sort(class.FeatureWeights,'descend');
    % taking only the specified number of features with the largest weights
    SelectedIdx = selected(1:Features2Select);
    disp('Features selected:');
    for ii = 1:length(SelectedIdx)
        disp(MIFeaturesName2D{1,SelectedIdx(ii)});
    end
    FeaturesTrainSelected = FeaturesTrain(:,SelectedIdx);       % updating the matrix feature
    FeaturesTest = FeaturesTest(:,SelectedIdx);                 % updating the matrix feature

    % visualize Feature Weights
    AllFeatureLabels = cellfun(@(x) regexp(x,': ','split','once'), MIFeaturesName2D(1,size(CSPFeatures,2)+1:end), 'UniformOutput', false);
    ElectrodeLabels = unique(cellfun(@(x) x{1}, AllFeatureLabels, 'UniformOutput', false), 'stable');
    FeatureTypeLabels = unique(cellfun(@(x) x{2}, AllFeatureLabels, 'UniformOutput', false), 'stable');
    FeatureWeights2D = reshape(class.FeatureWeights(size(CSPFeatures,2)+1:end), length(ElectrodeLabels), length(FeatureTypeLabels));
    figure; imagesc(FeatureWeights2D)
    set(gca, 'ytick', 1:length(ElectrodeLabels), 'yticklabels', ElectrodeLabels);
    set(gca, 'xtick', 1:length(FeatureTypeLabels), 'xticklabels', FeatureTypeLabels);
    xtickangle(45)
    colorbar
    c_lim = caxis;
    title('Feature Weights')
    if CSP_flag
        CSPFeatureLabels = cellfun(@(x) regexp(x,' ','split','once'), MIFeaturesName2D(1,1:size(CSPFeatures,2)), 'UniformOutput', false);
        figure; imagesc(class.FeatureWeights(1:size(CSPFeatures,2)));
        FeatureLabels = unique(cellfun(@(x) x{2}, CSPFeatureLabels, 'UniformOutput', false), 'stable');
        set(gca, 'ytick', 1:size(CSPFeatures,2), 'yticklabels', FeatureLabels);
        set(gca, 'xtick', 1, 'xticklabels', 'CSP');
        xtickangle(45)
        colorbar
        caxis(c_lim)
        title('CSP Feature Weights')
    end

    % save data
    save(strcat(recordingFolder,'\FeaturesTrain.mat'),'FeaturesTrain');
    save(strcat(recordingFolder,'\FeaturesTrainSelected.mat'),'FeaturesTrainSelected');
    save(strcat(recordingFolder,'\FeaturesTest.mat'),'FeaturesTest');
    save(strcat(recordingFolder,'\SelectedIdx.mat'),'SelectedIdx');
    relevantTrainFeatures = SelectedIdx;
    save(strcat(recordingFolder,'\relevantTrainFeatures.mat'),'relevantTrainFeatures');
    save(strcat(recordingFolder,'\LabelTest.mat'),'LabelTest');
    save(strcat(recordingFolder,'\LabelTrain.mat'),'LabelTrain');
    save(strcat(recordingFolder,'\CSP_flag.mat'),'CSP_flag');
    disp('Successfuly extracted features!');

    %% Feature selection with validation (choose best features across rep splits of data into train & test)
    rep = 100;
    SelectedIdx = zeros(rep, Features2Select);
    AllWeights = zeros(rep, size(MIFeatures,2));
    for ii = 1:rep
        testIdx = randperm(min_trials,num4test);                       % picking test index randomly   
        testIdx = [idleIdx(testIdx) leftIdx(testIdx) rightIdx(testIdx)];    % taking the test index from each class
        testIdx = sort(testIdx);                                            % sort the trials
        % split train data
        FeaturesTrain = MIFeatures;
        FeaturesTrain (testIdx ,:,:) = [];          % delete the test trials from the features matrix, and keep only the train trials
        LabelTrain = targetLabels;
        LabelTrain(testIdx) = [];                   % delete the test trials from the labels matrix, and keep only the train labels
        class = fscnca(FeaturesTrain,LabelTrain);   % feature selection
        AllWeights(ii,:) = class.FeatureWeights;
        % sorting the weights in desending order and keeping the indexs
        [~,selected] = sort(class.FeatureWeights,'descend');
        % taking only the specified number of features with the largest weights
        SelectedIdx(ii,:) = selected(1:Features2Select);
    end
    title('FSCNCA Feature Weights')
    figure; imagesc(AllWeights'); 
    xlabel('Train-Test splits')
    ylabel('Features')
    colorbar
    % check frequency of features
    num_features = size(FeaturesTrain,2);
    freqs = zeros(num_features,1);
    for ii = 1:num_features
       freqs(ii) = length(find(SelectedIdx(:) == ii));
    end
    freqs = freqs*100/rep;
    [freqs_sorted,selected] = sort(freqs,'descend');
    disp('Top features & their frequencies over many iterations:');
    for ii = 1:Features2Select
        disp([MIFeaturesName2D{1,selected(ii)} ': ' num2str(freqs_sorted(ii)) '%']);
    end
    % save top features of data
    AllDataTopFeaturesIdx = selected(1:Features2Select);
    AllDataTopFeatures = MIFeatures(:,AllDataTopFeaturesIdx);
    save(strcat(recordingFolder,'\AllDataTopFeaturesIdx.mat'),'AllDataTopFeaturesIdx');
    save(strcat(recordingFolder,'\AllDataTopFeatures.mat'),'AllDataTopFeatures');
end
end

