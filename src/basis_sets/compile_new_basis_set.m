%%% User defined input
% save_name = 'press_ge_30.mat'; % Must be unique so as not to overwrite other files
save_name = 'VERI_PRESS_30_GE_30_2000.mat';
template = '/home/john/Documents/Research/In-Vivo-MRSI-Simulator/src/basis_sets/PRESS_30_GE_2000.mat';
% new_path = '/home/john/Documents/Repositories/INSPECTOR/basis_sets/PRESS_30_GE_30_2000/IndividualSpins/*.mat';
new_path = '/home/john/Documents/Repositories/MARSSCompiled/GE_PRESS_30ms_w_MM/IndividualSpins/*.mat';
new_path = '/home/john/Documents/Repositories/MARSSCompiled/GE_PRESS_30ms_w_MM/SummedSpins_for_MARSSinput/*.mat';
new_path = '/home/john/Documents/Repositories/MARSSCompiled/VERI_GE_PRESS_30ms/SummedSpins_for_MARSSinput/*.mat';

sim_config = load('/home/john/Documents/Repositories/MARSSCompiled/MARSSInput.mat');
fn = fieldnames(sim_config);
B0 = sim_config.B0;
if ~ismember('referencePeak', fn)
    centerFreq = 4.6500;
else
    centerFreq = sim_config.referencePeak;
end

clear sim_config

% edit_off_paths = {}; % {'path/lactate.mat','path/gaba.mat'};
% metabs_off = {}; % {'lac','gaba'};

%%% Automated storage of metabolites and basis functions
% load(template)%, 'metabolites','header','artifacts')
fileinfo = dir(new_path);
fnames = {fileinfo.name};
directory = fileinfo(2,1).folder; %struct2cell(fileinfo); %directory
% directory = char(directory(2,1));
first = -1;
clear metabolites

for i=1:length(fnames)
    pth = fullfile(directory,fnames{i});
    load(pth,'exptDat')
    if first==-1
        dt = 1/exptDat.sw_h; 
        sw = exptDat.sw_h; 
        header.spectralwidth = sw;
        header.carrier_frequency = exptDat.sf; 
        header.Ns = exptDat.nspecC; 
        header.t = 0:dt:(dt*(header.Ns-1)); 
        header.centerFreq = centerFreq;
        header.B0 = B0;
        header.ppm = (-.5*sw:sw/(length(header.t)-1):0.5*sw) + centerFreq;
%         header.ppm = header.ppm + centerFreq;
        first = 1;
    end
    [a, metab,b] = fileparts(pth);
    metab = lower(metab);
    metabolites.(metab).fid(:,1,:) = real(exptDat.fid'); 
    metabolites.(metab).fid(:,2,:) = imag(exptDat.fid');
end

% Reordered the metabolties for the VERI basis set
% idx = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,25,26,27,28,29,30,31,32,15,16,17,18,19,20,21,22,23,24];
% metabolites = orderfields(metabolites,idx);
metabolites = reorder_metabolite_struct(metabolites);


%% 
ppm = -.5*header.spectralwidth/header.carrier_frequency:header.spectralwidth/header.carrier_frequency/(header.Ns - 1):0.5*header.spectralwidth/header.carrier_frequency; 
ppm = ppm + header.centerFreq; 
[min(ppm), max(ppm)]
% ppm = flip(ppm);

fn = fieldnames(metabolites);
tmp = cellfun(@(f) metabolites.(f).fid, fn, 'UniformOutput', false);

fn = fieldnames(metabolites);
num = numel(fn);

try
    stacked = cat(1, tmp{:});
catch
    stacked = zeros(num, 2, 8192);
    
    for k = 1:num
        fid = metabolites.(fn{k}).fid;
        fid = reshape(fid, [1, 2, 8192]);  % force shape
        stacked(k,:,:) = fid;
    end
end
fid = stacked(:,1,:) + 1j*stacked(:,2,:);
spec = fftshift(fft(fid,[],3),3);

figure, hold on, for i=1:num, plot(ppm, squeeze(real(spec(i,:,:)))), end, hold off, xlim([0,5]), set(gca,'xdir','reverse')


figure, hold on, for i=1:num, plot(min(ppm):(max(ppm)-min(ppm))/8191:max(ppm),squeeze(real(spec(i,1,:)))), end, hold off

%%
% %%% Saving
% [path, ~, ~] = fileparts(template); % Stores new basis set with the others
% path = '/home/john/Documents/Repositories/MRS-Sim/ignore/src/basis_sets';
path = '/home/john/Documents/Repositories/MRS-Sim/src/basis_sets';
save(fullfile(path,save_name), 'metabolites','header');%,'artifacts')

%%

%%% If spectral editing or othe unique basis functions need to be stored
%%% with their original FIDs, that should be added here.
if ~isempty(edit_off_paths)
    for i=1:length(edit_off_paths)
        load(pth,'exptDat')
        [a, metab,b] = fileparts(pth);
        metab = lower(metab);
        metabolites.(metab).fid_OFF = [real(exptDat.exptDat.fid'); imag(exptDat.exptDat.fid')];
    end
end







%%
%%% User defined input
% save_name = 'press_ge_30.mat'; % Must be unique so as not to overwrite other files
new_path = '/home/john/Documents/Repositories/INSPECTOR/basis_sets/PRESS_30_GE_30_2000/IndividualSpins/*.mat';

%%% Automated storage of metabolites and basis functions
% load(template)%, 'metabolites','header','artifacts')
fileinfo = dir(new_path);
fnames = {fileinfo.name};
directory = struct2cell(fileinfo); %directory
directory = char(directory(2,1));

for i=1:length(fnames)
    pth = fullfile(directory,fnames{i});
    load(pth,'exptDat')
%     data = exptDat.fid;
    data = fftshift(fft(exptDat.fid,[],1),1);
    s = size(data);
    mx = max(real(data),[],1);
    [ind, ignore] = find(real(data)==mx);
    stored = {fnames{i}};
    for ii=1:s(2)
%         f = figure;
%         plot(ppm, squeeze(real(data(:,ii)')))
%         set(gca,'xdir','reverse')
%         title(fnames{i})
        
%         for k=1:length(ind)
        stored(k+1) = {ppm(ind(ii))};
%         end
%         stored
%         waitforbuttonpress
%         close(f)
    end
    stored
end

%%
function S = reorder_metabolite_struct(S)
% Reorder struct fields alphabetically, then move MM* to the bottom
% in numeric order, then move Lip* to the very bottom in numeric order.
% Works recursively on nested structs.

    if ~isstruct(S)
        return
    end

    % Handle struct arrays elementwise
    if numel(S) > 1
        for n = 1:numel(S)
            S(n) = reorder_metabolite_struct(S(n));
        end
        return
    end

    fn = fieldnames(S);
    fnLower = lower(fn);

    isMM  = startsWith(fnLower, 'mm');
    isLip = startsWith(fnLower, 'lip');
    isOther = ~isMM & ~isLip;

    otherFields = sort(fn(isOther));
    mmFields    = sort_special_fields(fn(isMM),  'mm');
    lipFields   = sort_special_fields(fn(isLip), 'lip');

    newOrder = [otherFields; mmFields; lipFields];

    % Reorder current struct
    S = orderfields(S, newOrder);

    % Recurse into nested structs
    for k = 1:numel(newOrder)
        f = newOrder{k};
        val = S.(f);
        if isstruct(val)
            S.(f) = reorder_metabolite_struct(val);
        end
    end
end

function out = sort_special_fields(in, prefix)
% Sort fields like MM092, MM126, Lip003 by the numeric suffix.

    if isempty(in)
        out = in;
        return
    end

    nums = inf(numel(in), 1);

    for k = 1:numel(in)
        tok = regexp(in{k}, ['^' prefix '(\d+)'], 'tokens', 'once', 'ignorecase');
        if ~isempty(tok)
            nums(k) = str2double(tok{1});
        end
    end

    [~, idx] = sortrows([nums, (1:numel(in))']);
    out = in(idx);
end