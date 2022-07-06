function play_gifs(gifImages, cmaps, duration, hAxs, hFig)
% plays a gif for specified duration

num_gifs = length(gifImages);
numFrames = zeros(length(gifImages));
for ii = 1:num_gifs
    [~, ~, ~, numFrames(ii)] = size(gifImages{ii});
end
counter = 1;
start = tic;
while toc(start) < duration
    for ii = 1:num_gifs
        frame = mod(counter, numFrames(ii));
        if frame == 0, frame = numFrames(ii); end
        thisFrame = gifImages{ii}(:,:,:, frame);
        thisRGB{ii} = uint8(255 * ind2rgb(thisFrame, cmaps{ii}));
    end
    for ii = 1:num_gifs
        axes(hAxs(ii))
        imshow(thisRGB{ii});           
    end
%         drawnow;
%         pause(0.1)
    counter = counter + 1;
end
for ii = 1:num_gifs
   axes(hAxs(ii))
   cla
end
axes(hFig)

end