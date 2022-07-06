function [dataVar] = sortElectrodes(dataVar, EEG_data, EEG_event, startIndex, endIndex, numChans, trial)
%% sortElectrodes segments the data according to startIndex and endIndex

% dataVar = over-arching main data storage structure
% EEG_data = as outputed by the preprocessing stage
% EEG_events = event markers used to segment the data
% Fs = sample rate (used to transform time to sample points)
% startIndex = index of start of segment
% endIndex = index of end of segment
% numChans = total number of channels
% trial = specific trial index

for channel=1:numChans  
    data = EEG_data(channel, EEG_event(startIndex).latency : EEG_event(endIndex).latency);
    indices = 1:length(data);
    dataVar(trial,channel,indices) = data(indices); 
    if length(data) < size(dataVar,3)
        dataVar(trial,channel,indices(end)+1:size(dataVar,3)) = NaN;
    end
end


