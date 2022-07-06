function plot_accuracy(test_accuracy, chance)
    x_ticks = 1:1:length(test_accuracy);
    plot(x_ticks, test_accuracy, 'marker', '*', 'MarkerFaceColor', 'auto');
    set(gca, 'xtick', x_ticks)
    yline(chance, 'k', 'lineStyle', '--')
    xlim([x_ticks(1)-0.05 x_ticks(end) + 0.05])
    ylim([0 101]);
    xlabel('Repetition');
    ylabel('Accuracy (%)');
    box off;
end

