%%% User defined input
save_name = 'press_ge_30.mat'; % Must be unique so as not to overwrite other files
template = '/home/john/Documents/Research/In-Vivo-MRSI-Simulator/src/basis_sets/press_ge_144.mat';
new_path = '/home/john/Documents/Repositories/MARSSCompiled/GE_PRESS_30ms/SummedSpins_for_MARSSinput/*.mat';
B0 = 3;
centerFreq = 4.6500;
edit_off_paths = {}; % {'path/lactate.mat','path/gaba.mat'};
metabs_off = {}; % {'lac','gaba'};

%%% Automated storage of metabolites and basis functions
load template metabolites header artifacts
fileinfo = dir(new_path);
fnames = {fileinfo.name};
dir = struct2cell(fileinfo); dir = char(dir(2,1));
first = -1;

for i=1:length(fnames)
    pth = fullfile(dir,fnames{i});
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
        header.ppm = (-.5*sw:sw/(length(header.t-1):0.5*sw) + centerFreq;
        first = 1;
    end
    [a, metab,b] = fileparts(pth);
    metab = lower(metab);
    metabolites.(metab).fid = [real(exptDat.exptDat.fid'); imag(exptDat.exptDat.fid')];
end

%%% If spectral editing or othe unique basis functions need to be stored
%%% with their original FIDs, that should be added here.
if ~isempty(edit_off_paths)
    for i=1:length(edit_off_paths}
        load(pth,'exptDat')
        [a, metab,b] = fileparts(pth);
        metab = lower(metab);
        metabolites.(metab).fid_OFF = [real(exptDat.exptDat.fid'); imag(exptDat.exptDat.fid')];
    end
end

%%% Saving
[path, x, x] = fileparts(template); % Stores new basis set with the others
save(fullfile(path,save_name), 'metabolites','header','artifacts')
