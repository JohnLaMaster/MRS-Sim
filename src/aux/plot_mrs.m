%% plot_mrs.m
%
% Visualize MR spectroscopy fitting results.
%
% Layout
% ------
%   Top panel  : real spectrum (black) + baseline (blue) + fit (red, opt.)
%   Middle rows: per-metabolite basis functions – broadened & amplitude-scaled
%   Bottom row : residual water (no broadening)
%
% Expected data files
% -------------------
% MAT_FILE  (simulation outputs)
%   ppm              [N_pts]
%   spectra          [batch, 2, N_pts]  or  [batch, N_pts] complex
%   baselines        same shape as spectra
%   residual_water   same shape as spectra
%   params           struct with fields:
%                      .asc, .asp, .ch, …, .lip20   each [1, batch] (amplitudes)
%                      .d   [batch, N_met]  Lorentzian decay (Hz)
%                      .g   [batch, N_met]  Gaussian decay   (Hz²)
%                      (all amplitude fields come before .d in the struct)
%
% BASIS_FILE  (basis functions + acquisition header)
%   metabolites.<key>.fid   [2, N_pts]   row-1 real, row-2 imag
%   header.t                [1, N_pts]   time vector (seconds)
%   (header lives in the BASIS file, not the simulation file)

% clear; close all; clc;

% ══════════════════════════════════════════════════════════════
%  USER SETTINGS
% ══════════════════════════════════════════════════════════════
matFile    = '/home/john/Documents/Repositories/MRS-Sim/dataset/VERI_sample/dataset_spectra_0.mat';
basisFile  = '/home/john/Documents/Repositories/MRS-Sim/src/basis_sets/PRESS_30_GE_2000.mat';
specIdx    = 1;                % 1-based batch index
ppmRange   = [5.0, 0.5];      % [high_ppm, low_ppm]
showFit    = false;
figWidthPx = 520;
savePath   = '/home/john/Documents/Repositories/MRS-Sim/dataset/VERI_sample/mrs_plot.png';  % set '' to skip saving
% ══════════════════════════════════════════════════════════════

metKeys = {'asc','asp','ch','cr','gaba','gln','glu','gpc', ...
           'gsh','lac','mi','naa','naag','pch','pcr','pe', ...
           'si','tau','mm09','mm12','mm14','mm17','mm20', ...
           'lip09','lip13','lip20'};

metLabels = {'Asc','Asp','Ch','Cr','GABA','Gln','Glu','GPC', ...
             'GSH','Lac','MI','NAA','NAAG','PCh','PCr','PE', ...
             'SI','Tau','MM09','MM12','MM14','MM17','MM20', ...
             'Lip09','Lip13','Lip20'};

nMet = numel(metKeys);

% ──────────────────────────────────────────────────────────────
%  1.  Load simulation data
% ──────────────────────────────────────────────────────────────
S = load(matFile);

ppm      = double(squeeze(S.ppm));           % [N_pts]
specData = toComplexMRS(S.spectra);          % [batch, N_pts] complex
blData   = toComplexMRS(S.baselines);
rwData   = toComplexMRS(S.residual_water);

specReal = real(specData(specIdx, :));       % [1, N_pts]
blReal   = real(blData  (specIdx, :));
rwReal   = real(rwData  (specIdx, :));

% Amplitudes: params.<key> is [1, batch]; pick specIdx for each metabolite.
% Stack in metKeys order to build aVec [1, nMet].
p    = S.params;
aVec = zeros(1, nMet);
for i = 1:nMet
    vals     = double(p.(metKeys{i}));   % [1, batch]
    aVec(i)  = vals(specIdx);
end

dVec = double(p.d(specIdx, :));   % [1, nMet]
gVec = double(p.g(specIdx, :));

% ──────────────────────────────────────────────────────────────
%  2.  Load basis FIDs + time vector
%      NOTE: header (incl. t) is stored in the BASIS file,
%            not in the simulation .mat file.
% ──────────────────────────────────────────────────────────────
B = load(basisFile);
t = double(B.header.t);   % [1, N_pts]  time vector (seconds)

nPts       = numel(ppm);
metSpectra = zeros(nMet, nPts);

for i = 1:nMet
    key    = metKeys{i};
    fidRaw = double(B.metabolites.(key).fid);   % [2, N_pts]
    fidCx  = fidRaw(1,:) + 1j * fidRaw(2,:);
    fidB   = applyBroadening(fidCx, dVec(i), gVec(i), t);
    sp     = real(fftshift(fft(fidB)));
    metSpectra(i,:) = sp * aVec(i);
end

% ──────────────────────────────────────────────────────────────
%  3.  Reconstructed fit
% ──────────────────────────────────────────────────────────────
fitReal = sum(metSpectra, 1) + blReal;

% ──────────────────────────────────────────────────────────────
%  4.  Figure layout (manual axes positions)
% ──────────────────────────────────────────────────────────────
% Height units: top panel = 5, each metabolite/water row = 1
topUnits   = 5;
totalUnits = topUnits + nMet + 1;

lm = 0.04;  rm = 0.16;  bm = 0.05;  tm = 0.01;
pW = 1 - lm - rm;
pH = 1 - bm - tm;
uH = pH / totalUnits;

figHeight = figWidthPx * (totalUnits / 11);
fig = figure('Color','w','Units','pixels', ...
             'Position',[80 80 figWidthPx figHeight]);

% Top panel
axTop = axes('Position', [lm, 1-tm-topUnits*uH, pW, topUnits*uH]);

% Metabolite panels
axMet = gobjects(1, nMet);
for i = 1:nMet
    yBot     = 1 - tm - topUnits*uH - i*uH;
    axMet(i) = axes('Position', [lm, yBot, pW, uH]);
    linkaxes([axTop, axMet(i)], 'x');
end

% Water panel
axWat = axes('Position', [lm, bm, pW, uH]);
linkaxes([axTop, axWat], 'x');

% ──────────────────────────────────────────────────────────────
%  5.  Top panel
% ──────────────────────────────────────────────────────────────
axes(axTop); hold on;
xline(0, 'Color', [0.8 0.8 0.8], 'LineWidth', 0.8);
plot(ppm, specReal, 'k',                   'LineWidth', 0.9, 'DisplayName', 'data');
if showFit
    plot(ppm, fitReal, 'r',                'LineWidth', 0.9, 'DisplayName', 'fit');
end
plot(ppm, blReal, 'Color', [0.2 0.4 0.9], 'LineWidth', 0.9, 'DisplayName', 'baseline');
legend('Location','northeast', 'FontSize',6.5, 'Box','off');
% xlim sets the visible window; the renderer clips everything outside automatically
set(axTop, 'XDir','reverse', 'XLim',[ppmRange(2), ppmRange(1)]);
cleanAxes(axTop);
hold off;

% ──────────────────────────────────────────────────────────────
%  6.  Metabolite panels
% ──────────────────────────────────────────────────────────────
for i = 1:nMet
    axes(axMet(i)); hold on;
    xline(0, 'Color', [0.8 0.8 0.8], 'LineWidth', 0.4);
    plot(ppm, metSpectra(i,:), 'k', 'LineWidth', 0.55);
    cleanAxes(axMet(i));
    text(1.01, 0.5, metLabels{i}, 'Units','normalized', ...
        'FontSize',6.5, 'VerticalAlignment','middle', ...
        'HorizontalAlignment','left');
    hold off;
end

% ──────────────────────────────────────────────────────────────
%  7.  Residual water panel
% ──────────────────────────────────────────────────────────────
axes(axWat); hold on;
xline(0, 'Color', [0.8 0.8 0.8], 'LineWidth', 0.4);
plot(ppm, rwReal, 'k', 'LineWidth', 0.55);
cleanAxes(axWat);
text(1.01, 0.5, 'H_2O', 'Units','normalized', ...
    'FontSize',6.5, 'VerticalAlignment','middle', ...
    'HorizontalAlignment','left');

% x-axis ticks on the water panel only
set(axWat, 'XTickLabelMode','auto', 'XColor','k', 'XAxis.Visible','on');
axWat.XAxis.TickValues      = ppmRange(2) : 0.5 : ppmRange(1);
axWat.XAxis.MinorTickValues = ppmRange(2) : 0.1 : ppmRange(1);
axWat.XMinorTick            = 'on';
axWat.TickLength            = [0.015, 0.008];
axWat.FontSize              = 8;
xlabel(axWat, 'Chemical Shift (ppm)', 'FontSize', 9);
hold off;

% ──────────────────────────────────────────────────────────────
%  8.  Save
% ──────────────────────────────────────────────────────────────
if ~isempty(savePath)
    exportgraphics(fig, savePath, 'Resolution', 180);
    fprintf('Saved → %s\n', savePath);
end


% ══════════════════════════════════════════════════════════════
%  LOCAL FUNCTIONS
% ══════════════════════════════════════════════════════════════

function c = toComplexMRS(arr)
% Convert to complex. If already complex → return as-is.
% Otherwise assume shape [batch, 2, N_pts] with dim-2 = [real; imag].
    arr = double(arr);
    if ~isreal(arr)
        c = arr;
        return;
    end
    try
        c = squeeze(arr(:,1,:) + 1j * arr(:,2,:));   % [batch, N_pts]
    catch
        c = squeeze(arr(:,1,1,:) + 1j * arr(:,1,2,:));   % [batch, N_pts]
    end
end


function fidOut = applyBroadening(fid, d, g, t)
% Lorentzian + Gaussian apodization:  fid · exp(−d·t − g·t²)
%
% d : Lorentzian decay rate (Hz, from params.d)
% g : Gaussian decay rate   (Hz², from params.g)
% t : time vector (seconds, from basis header.t)
    fidOut = fid .* exp(-d .* t - g .* t.^2);
end


function cleanAxes(ax)
% Remove spines, y-ticks, and x-tick labels.
    set(ax, 'YTick',[], 'XTickLabel',[], 'TickLength',[0 0], ...
        'XColor','none', 'YColor','none');
    box(ax, 'off');
end


% ══════════════════════════════════════════════════════════════
%  NOTE: Loading MATLAB v7.3 / HDF5 .mat files
% ══════════════════════════════════════════════════════════════
% v7.3 files cannot be opened with load(). Use matfile() or h5read():
%
%   m      = matfile(matFile);
%   ppm    = m.ppm;
%   % etc.
%
%   % For the basis header:
%   t      = h5read(basisFile, '/header/t');
%   fidRaw = h5read(basisFile, '/metabolites/asc/fid');   % [2 x N]