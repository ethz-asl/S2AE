% AVG Jaccard Coefficient
jaccard_fusion = [0.6507, 0.6504, 0.6598, 0.6545, 0.6606, 0.6709, 0.6688, 0.671, 0.6716, 0.6773];
jaccard_orig = [0.5045, 0.5409, 0.5528, 0.5772, 0.5813, 0.5938, 0.5964, 0.5964, 0.6008, 0.6113];


% AVG DICE Coefficient
dice_fusion = [0.7423, 0.7398, 0.7489, 0.7419, 0.7483, 0.76, 0.7587, 0.7583, 0.7578, 0.7666];
dice_orig = [0.5941, 0.6335, 0.6447, 0.6682, 0.6772, 0.6842, 0.6871, 0.6879, 0.6975, 0.701];
n = size(dice_fusion, 2);

% ACG Pixel Accuracy
px_fusion = [0.966, 0.9659, 0.9675, 0.9673, 0.9678, 0.9681, 0.969, 0.9695, 0.97, 0.9696];
px_orig = [0.9474, 0.9524, 0.9542, 0.9593, 0.9605, 0.9615, 0.9619, 0.9603, 0.9642, 0.9641];

% compute improvement
jaccard_diff = jaccard_fusion - jaccard_orig;
jaccard_improvement = jaccard_diff ./ jaccard_orig * 100;

dice_diff = dice_fusion - dice_orig;
dice_improvement = dice_diff ./ dice_orig * 100;

px_diff = px_fusion - px_orig;
px_improvement = px_diff ./ px_orig * 100;

clf;
plot(1:n, smoothdata(dice_improvement), 'DisplayName', 'DICE Coefficient');
hold on;
plot(1:n, smoothdata(jaccard_improvement), 'DisplayName', 'Jaccard Index');
plot(1:n, smoothdata(px_improvement), 'DisplayName', 'Pixel Accuracy');
grid on;
legend();
ylabel('\mathrm{Improvement}\,[\%]')
xlabel('\mathrm{Epoch}')
xticks(1:n);
xlim([1 n])
beautifySimplePlot('Fusion Improvement');
prepareFig();
set(gcf,'color','w');

function beautifySimplePlot(name) 
      set(gca         , ...
        'FontName'    , 'Helvetica' );
      
      set(gca         , ...
        'FontSize'    , 10           );
      
      set(gca, ...
        'Box'         , 'off'     , ...
        'TickDir'     , 'out'     , ...
        'TickLength'  , [.02 .02] , ...
        'XMinorTick'  , 'off'      , ...
        'YMinorTick'  , 'on'      , ...
        'YGrid'       , 'on'      , ...
        'XGrid'       , 'on'      , ...
        'XColor'      , [.3 .3 .3], ...
        'YColor'      , [.3 .3 .3], ...        
        'LineWidth'   , 1.5         );
      title(name);
end  