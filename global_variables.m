global gifs_folder;
gifs_folder = 'LabelImages\';

global numClasses idleIndex leftIndex rightIndex;
numClasses = 3;
idleIndex = 1;
leftIndex = 2;
rightIndex = 3;

global startRecordings startTrial Baseline endTrial endRecording;
startRecordings = 000;          
startTrial = 1111;
Baseline = 1001;
endTrial = 9;
endRecording = 99;

global EEG_chans_names;
EEG_chans_names = {'C03', 'C04', 'C0Z', 'FC1', 'FC2', 'FC5', 'FC6', 'CP1', 'CP2', 'CP5', 'CP6', 'O01', 'O02'};

% all paths necessary to run code - change per computer
paths;


