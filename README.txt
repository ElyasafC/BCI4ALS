# BCI4ALS- Team 35:

## README

This is the code repository for the BCI4ALS, team 35.
The code is a fork of Asaf Harel(harelasa@post.bgu.ac.il) basic code for the course BCI-4-ALS which
took place during 2021-2022. You are free to use, change, adapt and
so on - but please cite properly if published.

In this file we will explain the:
1. Set up and installation requirements.
2. Code parts and how to run it all.
3. Additional files and recordings.


1. Set up and installation requirements:
- python 3.8 and up.
- MATLAB 2021b and up (for the function 'pyrunfile')
	- MATLAB toolboxes: 
		1. Statistics and Machine Learning Toolbox
		2. Signal Processing Toolbox
		3. Mapping Toolbox
- Lab Streaming Layer 1.14.0. 
- LabRecorder  1.14.2.
- EEGLAB
- ERPLAB 8.20.
- LoadXDF (XDF import) via EEGLAB’s plugins manager.
- OpenBCI GUI 5.0.8.


2. Code parts and how to run it all:


Main files:
- MI1_offline_training.m - Main code for the offline recording.
- MI2_preprocess.m - Function to preprocess raw EEG data.
- MI3_segmentation.m - Function that segments the preprocessed data into chunks.
- MI4_featureExtraction.m - Function to extract features from the segmented chunks.
- MI5_classifier.py - Classifier class to train a model and predict new datapoints.
- MI_OnlineFeedback.m - Main code to run with trained model on live recording.

Subfiles:
- Functions that are called through running the main files.
- global_variables.m - defines values of variables used across files. Need to make sure the EEG channel names (EEG_chans_names) matches the channels on the helmet that you are using. 
- paths.m - paths to toolboxes, files and data. Needs to be updated to personal computer.
- saveProblemTrials.m - saves indices of trials that were problematic during the recording session. If relevant, run this before MI3_segmentation.m.
- combineRecordingsTrials.m - code to combine trials from different recording sessions into one database. If desired, run this before MI4_featureExtraction.m.
- removeNoisyTrials.m - removes trials in which some electrode was very high or very low in amplitude. If desired, run this before MI4_featureExtraction.m.
- MI_CoAdaptiveLearning.m - Main code to run recording with trained model in which both the model is updated throughout the recording and the subject is given live feedback and can learn to adapt to the model. This code still needs to be optimized.   
- incrementalMulticlassModel.m - model which has the option of incremental learning. This is used for the MI_CoAdaptiveLearning.m.
 
Additional files:
- active6 - hardware settings of the helmet. Needs to be used in the OpenBCI setup.
- .ced files describe the EEG channels locations on the helmet. These are required in MI2_preprocess.m if ICA_flag = 1. 
	- montage_ultracortex_10_chans_(no_CP5,O1,O2).ced - 10 channels: {'C03', 'C04', 'C0Z', 'FC1', 'FC2', 'FC5', 'FC6', 'CP1', 'CP2', 'CP6'}
	- montage_ultracortex_10_chans_(no_FC2,O1,O2).ced - 10 channels: {'C03', 'C04', 'C0Z', 'FC1', 'FC5', 'FC6', 'CP1', 'CP2', 'CP5', 'CP6'}
	- montage_ultracortex_11_chans.ced - 11 channels: {'C03', 'C04', 'C0Z', 'FC1', 'FC2', 'FC5', 'FC6', 'CP1', 'CP2', 'CP5', 'CP6'}
	- montage_ultracortex_13_chans.ced - 11 channels: {'C03', 'C04', 'C0Z', 'FC1', 'FC2', 'FC5', 'FC6', 'CP1', 'CP2', 'CP5', 'CP6', 'O01', 'O02'}
- .jpg and .gif files are used in MI1_offline_training.m and MI_OnlineFeedback.m. 



### Offline 

1. Arrange the helmet.
2. Open OpenBCI.
3. Open LabRecorder  
4. Start MI1

(*** More details in additional file 'שלבים להקלטות' https://docs.google.com/document/d/1TzT4basx_MSigC5rpA7EyLNnI_y45cA9m-mObJj53rQ/edit
     and in additional file 'שלבים לפגישות' https://docs.google.com/document/d/1l3p0VU2F4rlwTHMI1Qyp_RLF6tRu7XGJ8jxEoUls73A/edit***)


PreProcess, analyze and train model:

1. Run MI2: analyze graphes and make changes to preprocessing if necessary.
2. Run saveProblemTrials.m if relevant 
3. Run MI3
4. Option to remove noisy trials (removeNoisyTrials.m) or/and combining recorded datasets (combineRecordingsTrials.m).
5. Run MI4: analyze graphs and make changes to feature extraction if desired.
6. Train model with MI_classifier.

(*** More details in additional file -'Analysis steps' https://docs.google.com/document/d/121vz3AndphhoJDgTiciF8R20hagNE7StI_OymxUSFUQ/edit***)


### Online

Same set up as in offline recording but run MI_OnlineFeedback instead of MI1_offline_training. 
Need to make sure to first have existing data on which MI1-MI4 were run. 
MI_OnlineFeedback will begin by training a model on this data. You can change the parameters values of this model in lines 47-48 (see MI5_classifier.py for more details).

### Python

First need to install all the requirments packages.


## Recordings

The general structure of the recordings directory are as follows:

1. Main recording directory consists of folders for each recording day.
   The folders named by the dates are the recordings of our mentor. There is one folder of a recording of us.
2. In each subfolder there are generally three subfolders, one for each session of the day. A short note on each day
   explains problems during the session if they exist.
3. In each session folder, there is a folder with the raw data, and folders of the processed data: one using all 11 electrodes 
   and one without problematic ones.


***

For more info, see the documentation located in each code file and the docs file in the documents folder.

### Trobuleshooting

### The dongle is plugged in but cannot connect to the headset.
1. Try replacing the batteries.

### Nothing works :(
1. Disconnect the dongle.
2. Close all programs.
3. Wait a bit!
4. Try again. :)

✨  Contact details  ✨

- Nitai Seri(nitai.seri@mail.huji.ac.il)
- Chaviva Moshavi (chaviva.moshavi@mail.huji.ac.il)
- Elyasaf Cohen (elyasaf.cohen@mail.huji.ac.il)
