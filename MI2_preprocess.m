function [EEG_data] = MI2_preprocess(recordingFolder, ICA_flag, EEG_data, Fs, EEG_chans, ICA_info)
%% Offline Preprocessing
% Assumes recorded using Lab Recorder.
% Make sure you have EEGLAB installed with ERPLAB & loadXDF plugins.

% For OFFLINE preprocessing:
% recordingFolder - where the EEG (data & meta-data) are stored.
% Assumes the EEG file is called EEG.xdf

% ICA_flag: 1 or 0, if not entered, ICA_Flag will be set to 0

% Adaptation for ONLINE preprocessing:
% Instead of "recordingFolder", need the following parameters:
% (1) EEG_data (data of electrodes) 
% (2) Fs (sampling  rate)
% (3) EEG_chans (names of relevant channels)
% (4) ICA_info (struct with ICA weights and information) - required if ICA_flag is 1

% Preprocessing using EEGLAB function.
% 1. load XDF file (Lab Recorder LSL output)
% 2. filter data above 0.5 & below 40 Hz
% 3. notch filter @ 50 Hz
% 4. average re-referencing
% 5. advanced artifact removal: ASR
% 6. advanced artifact removal: option for ICA

%% This code is part of the BCI-4-ALS Course written by Asaf Harel
% (harelasa@post.bgu.ac.il) in 2021. You are free to use, change, adapt and
% so on - but please cite properly if published.
% This code was adapated by HUJI team 35 as part of the BCI-4-ALS course
%%

%%
global_variables
global eeglab_path
%% Parameters
if nargin < 2 || isempty(ICA_flag)
    ICA_flag = 0;
end
highLim = 40;                               % filter data under 40 Hz
lowLim = 0.5;                               % filter data above 0.5 Hz
addpath(genpath(eeglab_path))               % update eeglab_path in paths.m to your own computer path

%% Load EEG data
if ~isempty(recordingFolder)

    % Offline processing
    offline = true;
    eeglab;                                     % open EEGLAB 
    recordingFile = strcat(recordingFolder,'\EEG.xdf');

    % (1) Load subject data (assume XDF)
    EEG = pop_loadxdf(recordingFile, 'streamtype', 'EEG', 'exclude_markerstreams', {});
else 
    % Online processing - get EEG data itself
    offline = false;
    if nargin < 3 || isempty(EEG_data)
       disp('Please enter EEG data')
       return
    elseif nargin < 4 || isempty(Fs) 
       disp('Please enter sampling rate')
       return
    elseif nargin < 5 || isempty(EEG_chans)
        disp('Please enter sampling rate')
        return   
    elseif nargin < 6 || isempty(ICA_flag)
        if ICA_flag
            disp('Please enter ICA_info')
            return
        else
            ICA_info = [];
        end
    end
    EEG.data = EEG_data;
    EEG.srate = Fs;
    EEG.trials = 1;
    EEG.filename = '';
    EEG.filepath = '';
    EEG.history = '';
    EEG.epoch = [];
    EEG.xmin = 0;
    EEG.xmax = 1.1882e+03;
    EEG.event = [];
    EEG.pnts = size(EEG.data,2);
end
EEG.setname = 'MI_sub';

%% Select channels
global EEG_chans_names
if offline
    EEG_chans_idx_use = 1:length(EEG_chans_names);

    % Define channels that we don't want to use or that are noisy
    % Channels that will be removed altogether
    EEG_chans_idx_no_use = [12,13]; 
    % NOISY CHANNELS THAT WE ARE *NOT* REMOVING
    % NOTE THAT THE INDEX WILL CHANGE IF YOU REMOVE AN ELECTRODE SMALLER
    % THAN THE NOISY ELECTRODES
    EEG_chans_idx_noisy = [];

    % Remove channels
    EEG_chans_idx_use(EEG_chans_idx_no_use) = [];
    for ii = 1:length(EEG_chans_idx_use)
        EEG_chans(ii,:) = EEG_chans_names{EEG_chans_idx_use(ii)};    
    end  
    EEG.data = EEG.data(EEG_chans_idx_use,:);

    % Determine noisy channels
    non_noisy_chans = 1:size(EEG_chans,1);
    non_noisy_chans(EEG_chans_idx_noisy) = [];
else
    % Online - 
    % 1. EEG_chans has all selected channels already
    % 2. For online preprocessing we don't identify noisy channels
    non_noisy_chans = 1:size(EEG_chans,1);
end
EEG.nbchan = size(EEG_chans,1);

%% Plot raw data
if offline
    figure; hold on;
    [~, exp_timeS] = min(abs(EEG.times - EEG.event(2).latency/(EEG.srate)*1000));
    [~, exp_timeL] = min(abs(EEG.times - EEG.event(end-1).latency/(EEG.srate)*1000));
    EEG_time_exp = EEG.times(exp_timeS:exp_timeL)-EEG.times(exp_timeS);
    time_min = EEG_time_exp./(1000*60);
    for ii = 1:size(EEG_chans,1)    
        plot(time_min, EEG.data(ii,exp_timeS:exp_timeL));   
    end
    xlabel('Time (min)')
    xlim([time_min(1) time_min(end)])
    %% Plot raw data
    eegplot(EEG.data)
    %% PSD 
    figure;
    [spectra,freqs] = spectopo(EEG.data,0,EEG.srate,'percent', 50,'freqrange',[0, 60],'electrodes','off');
    legend(EEG_chans);
    title('Raw data')
end

%% Filter data 
% Low-pass filter
EEG = pop_eegfiltnew(EEG, 'hicutoff',highLim,'plotfreqz',0);    
if offline
    EEG = eeg_checkset( EEG );
    figure;
    [spectra,freqs] = spectopo(EEG.data,0,EEG.srate,'percent', 50,'freqrange',[0, 60],'electrodes','off');
    title('Low pass filter')
    legend(EEG_chans);  
end
% High-pass filter
EEG = pop_eegfiltnew(EEG, 'locutoff',lowLim,'plotfreqz',0);  
if offline
    EEG = eeg_checkset( EEG );
    figure;
    [spectra,freqs] = spectopo(EEG.data,0,EEG.srate,'percent', 50,'freqrange',[0, 60],'electrodes','off');%, limits,title,freqfaq, percent);
    title('High pass filter')
    legend(EEG_chans);    
end

% Notch filter - this uses the ERPLAB filter
EEG  = pop_basicfilter( EEG,  1:EEG.nbchan, 'Boundary', 'boundary', 'Cutoff',  50, 'Design', 'notch', 'Filter', 'PMnotch', 'Order',  180 );
if offline
    EEG = eeg_checkset( EEG );
    figure;
    [spectra,freqs] = spectopo(EEG.data,0,EEG.srate,'percent', 50,'freqrange',[0, 60],'electrodes','off');%, limits,title,freqfaq, percent);
    title('Notch filter')
    legend(EEG_chans);   
end

%% Re-referencing (subtracting the mean)
if offline
    eegplot(EEG.data)
    % sanity check - check that all channels are on the same magnitude
    figure; hold on;
    for chan = 1:EEG.nbchan
       histogram(EEG.data(chan,:), 'DisplayName', EEG_chans(chan,:));
    end
    legend()
end

% subtract the mean of the non-noisy channels
EEG.data = EEG.data  - mean(EEG.data(non_noisy_chans,:),1);

if offline
    eegplot(EEG.data)
end

%% ASR - basic cleaning
if offline
    EEG = pop_clean_rawdata(EEG, 'FlatlineCriterion','off','ChannelCriterion','off','LineNoiseCriterion','off','Highpass','off','BurstCriterion',20,'WindowCriterion','off','BurstRejection','off','Distance','Euclidian');
    eegplot(EEG.data)
end

%% ICA - optional
%% Choose data to train ICA on
if ICA_flag 
    if offline
        window = 20;
        ii = 1;
        RMS = zeros(EEG.nbchan,ceil(length(EEG.data)/window));
        jj = 1;
        while ii < length(EEG.data)
           d = EEG.data(:,ii:min(length(EEG.data), ii+window-1));
           for chan = 1:EEG.nbchan
               RMS(chan,jj) = sqrt(mean((d(chan,:) - mean(d(chan,:))).^2));
           end
           ii = ii + window;
           jj = jj + 1;
        end
        figure; rows = round(sqrt(EEG.nbchan)); cols = ceil(EEG.nbchan/rows);
        windows_to_remove_chan = zeros(size(RMS)); 
        for chan = 1:EEG.nbchan
            subplot(rows,cols,chan)
            histogram(RMS(chan,:)); 
            title(EEG_chans(chan,:))
            sd = std(RMS(chan,:));
            windows_to_remove_chan(chan, RMS(chan,:) > 4*sd) = 1;
        end
        sgtitle('RMS distributions')
        windows_to_remove = any(windows_to_remove_chan,1);
        samples_to_keep = zeros(1,length(EEG.data));
        ii = 1; jj = 1;
        while ii <= length(EEG.data)
            if windows_to_remove(jj) == 0
               samples_to_keep(min(length(EEG.data), ii:ii+window-1)) = 1;
            end
           ii = ii + window; jj = jj + 1;
        end
        EEG_4ICA = EEG;
        EEG_4ICA.data = EEG.data(:,logical(samples_to_keep));
        %% Perform ICA
        ICA_info = train_ICA(EEG_4ICA, EEG_chans);
    end
    if ~isempty(ICA_info)
        %% Remove non-brain ICA components
        brain_label = find(strcmp(ICA_info.etc.ic_classification.ICLabel.classes, 'Brain'));
        non_brain_components = find(ICA_info.etc.ic_classification.ICLabel.classifications(:,brain_label) < 0.4);
        % remove these components
        components_to_remove = [non_brain_components];
        EEG.icaweights = ICA_info.icaweights;
        EEG.icawinv = ICA_info.icawinv;
        EEG.icasphere = ICA_info.icasphere;
        EEG.icachansind = ICA_info.icachansind;
        EEG = pop_subcomp(EEG, components_to_remove, 0);
        %% Run ICA again on data if desired
        % train_ICA(EEG);
%         eegplot(EEG.data)
    end
else
    ICA_info = [];
end
%% Save data
if offline
    % Save the data into .mat variables on the computer
    EEG_data = EEG.data;            % Pre-processed EEG data
    EEG_event = EEG.event;          % Saved markers for sorting the data
    save(strcat(recordingFolder,'\','cleaned_sub.mat'),'EEG_data');
    save(strcat(recordingFolder,'\','EEG_events.mat'),'EEG_event');
    save(strcat(recordingFolder,'\','EEG_chans.mat'),'EEG_chans');
    save(strcat(recordingFolder,'\','EEG_chans_idx_noisy.mat'),'EEG_chans_idx_noisy');
    if ~isempty(ICA_info)
        ICA_info.etc.ic_classification.ICLabel = ICA_info.etc.ic_classification.ICLabel;
        ICA_info.icaweights = ICA_info.icaweights;
        ICA_info.icawinv = ICA_info.icawinv;
        ICA_info.icasphere = ICA_info.icasphere;
        ICA_info.icachansind = ICA_info.icachansind;
        save(strcat(recordingFolder,'\','ICA_info.mat'),'ICA_info');
    end
else
   EEG_data = EEG.data; 
end

end
