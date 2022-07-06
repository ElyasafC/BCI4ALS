function [] = plotEstimate(negloss_combine, labels, images, hAx)
%% This function plots a bar graph with the estimated prediction 

% draw prediction
axes(hAx) 
hold on
class = [1, 2, 3];
pred = class(negloss_combine == max(negloss_combine));
img = images{pred};
if pred == 1
    add = 0;
elseif pred == 2
    add = 0.15;
else
    add = 0.13;
end
image(flip(img, 1),  'XData', [0.3 0.52] + add, ...
    'YData', [0.6 1.45 * size(img,1) ./ size(img,2)])

text(0.5 ,0.9 , 'The estimated prediction is: ', ...
     'HorizontalAlignment', 'Center', 'Color', 'white', 'FontSize', 24);

ax1 = axes('Position',[.3 .2 .4 .4]);
set(ax1,'color', 'black');
b = bar(negloss_combine);
b.FaceColor = 'flat';
set(gca, 'visible', 'off');
xtips = b(1).XEndPoints;
ytips = b(1).YEndPoints;
b.CData(pred, :) = [.1, .6, .1];
text(xtips, ytips, labels, 'HorizontalAlignment', 'center', ...
    'verticalAlignment', 'top', 'color', 'w', 'FontSize', 17)


set(ax1,'color', 'black'); 

end