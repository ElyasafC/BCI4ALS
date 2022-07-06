function EEG = train_ICA(EEG, EEG_chans)
    global_variables
    EEG = pop_runica(EEG, 'icatype', 'runica', 'extended',1,'interrupt','on');
    % add channel locations
    if EEG.nbchan == 13
        chanlocfile = chan13_loc_file;
    elseif EEG.nbchan == 11
        chanlocfile = chan11_loc_file;        
    elseif EEG.nbchan == 10 && ~any(sum(EEG_chans == 'FC2',2) == 3)
        chanlocfile = chan10NoFC2_loc_file;                
    elseif EEG.nbchan == 10 && ~any(sum(EEG_chans == 'CP5',2) == 3)
        chanlocfile = chan10NoCP5_loc_file; 
    end
    EEG=pop_chanedit(EEG, 'lookup',standard_1005_path,'load',{chanlocfile,'filetype','autodetect'});
    EEG=pop_chanedit(EEG, 'lookup',standard_1005_path);
    % add ICA labels
    EEG = pop_iclabel(EEG, 'default');
    % find components that are less than 50% brain activity
    figure; imagesc(EEG.etc.ic_classification.ICLabel.classifications);
    set(gca, 'xtick', 1:length(EEG.etc.ic_classification.ICLabel.classes), 'xticklabels', EEG.etc.ic_classification.ICLabel.classes);
    xtickangle(45)
    colorbar
    title('ICA')
end

