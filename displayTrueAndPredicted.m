function [] = displayTrueAndPredicted(true, predicted, labels, images)


img_xdata = [0.25, 0.75];

img = images{true};

image(flip(img, 1), 'XData', img_xdata,...
    'YData', [0.5, 1.35 * size(img,1)./ size(img,2)])

text(0.5 ,0.8 , '\underline{The true class is:}', ...
     'HorizontalAlignment', 'Center', 'Color', 'white', 'FontSize', 24, 'Interpreter', 'latex');
text(0.5 ,0.74 , labels{true}, ...
     'HorizontalAlignment', 'Center', 'Color', 'white', 'FontSize', 24, 'FontWeight', 'Bold');
  

img = images{predicted};

image(flip(img, 1), 'XData', img_xdata,...
    'YData', [0.1, 0.65 * size(img,1)./ size(img,2)])

text(0.5 ,0.42 , '\underline{The predicted class is:}', ...
     'HorizontalAlignment', 'Center', 'Color', 'white', 'FontSize', 24, 'Interpreter', 'latex');
text(0.5 ,0.345, labels{predicted}, ...
     'HorizontalAlignment', 'Center', 'Color', 'white', 'FontSize', 24, 'FontWeight', 'Bold');



end

