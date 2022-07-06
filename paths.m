
%% Paths - change per computer

% lab streaming layer
lsl_lib_path = 'C:\Toolboxes\liblsl-Matlab-1.14.0-Win_amd64_R2020b\liblsl-Matlab\';
lsl_bin_path = 'C:\Toolboxes\liblsl-Matlab-1.14.0-Win_amd64_R2020b\liblsl-Matlab\bin';

% eeglab
eeglab_path = 'C:\Toolboxes\EEGLAB';

% files for channel locations for ICA
chan13_loc_file = 'C:\\Users\\chubb\\Documents\\BCI4ALS\\montage_ultracortex_13_chans.ced'; % with all electrodes
chan11_loc_file = 'C:\\Users\\chubb\\Documents\\BCI4ALS\\montage_ultracortex_11_chans.ced'; % without electrodes O1,O2
chan10NoFC2_loc_file = 'C:\\Users\\chubb\\Documents\\BCI4ALS\\montage_ultracortex_10_chans_(no_FC2,O1,O2).ced'; % without electroes FC2, O1, O2
chan10NoCP5_loc_file = 'C:\\Users\\chubb\\Documents\\BCI4ALS\\montage_ultracortex_10_chans_(no_CP5,O1,O2).ced'; % without electroes FC2, O1, O2
standard_1005_path = 'C:\\Toolboxes\\EEGLAB\\plugins\\dipfit\\standard_BEM\\elec\\standard_1005.elc';

% data directories
dataTopDir = 'C:\Users\chubb\Documents\BCI4ALS_data\Mentor_data\'; % top directory with recording folders
recordingFolders = {
    [dataTopDir '27.3\first\electrodes 1-11\']; % 1
    [dataTopDir '11.4\first\electrodes 1-11\']; % 2
    [dataTopDir '11.4\second\electrodes 1-11\'];% 3
    [dataTopDir '11.4\third\electrodes 1-11\']; % 4
    [dataTopDir '24.4\first\electrodes 1-11\']; % 5
    [dataTopDir '24.4\second\electrodes 1-11\'];% 6
    [dataTopDir '24.4\third\electrodes 1-11\']; % 7
    [dataTopDir '1.5\first\electrodes 1-11\'];  % 8
    [dataTopDir '1.5\second\electrodes 1-11\']; % 9
    [dataTopDir '1.5\third\electrodes 1-11\'];  % 10
    [dataTopDir '22.5\first\electrodes 1-11\']; % 11
    [dataTopDir '22.5\second\electrodes 1-11\'];% 12
    [dataTopDir '22.5\third\electrodes 1-11\']; % 13
    [dataTopDir '8.6\first\electrodes 1-11\'];  % 14
    [dataTopDir '8.6\Online2\'];  % 15
};





