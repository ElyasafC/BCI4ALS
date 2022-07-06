function [gifImage, gifCmap] = select_random_idle_gif(Folder, FileList)
% selects random gif from folder

    Index = randperm(numel(FileList), 1);
    Source = fullfile(Folder, FileList(Index).name);
    [gifImage, gifCmap] = imread(Source, 'gif', 'Frames', 'all');
        
end