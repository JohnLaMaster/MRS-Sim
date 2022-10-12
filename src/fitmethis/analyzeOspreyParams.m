function dist,parameters = analyzeOspreyParams(path)
% Takes in the path of the .mat file produced by osp_exportParams.m in Osprey.
% Those variables are then compiled and analyzed using fitmethis to determine
% their distribution and the defining parameters.
% 
%     - header      [metadata]
%     - ampl        
%     - ph0         [deg]
%     - ph1         [deg/ppm]
%     - gaussLB     [Hz]
%     - lorentzLB   [Hz]
%     - freqShift   [Hz]
%     - refShift    [Hz]
% =========+=========+=========+=========+=========+=========+=========+=========+
params = load(path)
num_bF = params.header.nMets + params.header.nMM;
names = {'ampl','lorentzLB','gaussLB','refShift','freqShift','ph0','ph1','SNR'};
len = [0, num_bF, num_bF, 1, 1, num_bF, 1, 1, 1];
clen = cumsum(len);
ind = {}
for i=2:length(len)
    ind{i-1} = (1:len(i)) + clen(i-1);
end

fields = fieldnames(params);
fields(strcmp(fields,'header')) = [];
fields(strcmp(fields,'ECC')) = [];


sz = [params.header.nDatasets, (2*num_bF + 2 + 1 + num_bF + 3)];
parameters = zeros([params.header.nDatasets, sz]);
clen = clen + 1;

for i=1:length(names)
    parameters(:,clen(i:i+1)) = params.(names{i});
end

variable_names = params.header.names;
for i=1:length(variable_names)
    dist.(variable_names{i}).ampl.F = fitmethis(parameters(:,i),'dtype','cont',
                                                'criterion','LL'){1};
    dist.(variable_names{i}).ampl.min = min(squeeze(parameters(:,i)));
    dist.(variable_names{i}).ampl.max = max(squeeze(parameters(:,i)));
    dist.(variable_names{i}).d.F = fitmethis(parameters(:,i+num_bF),'dtype','cont',
                                             'criterion','LL'){1};
    dist.(variable_names{i}).d.min = min(squeeze(parameters(:,i+num_bF)));
    dist.(variable_names{i}).d.max = max(squeeze(parameters(:,i+num_bF)));
    dist.(variable_names{i}).fshift.F = fitmethis(parameters(:,i+clen(5)),'dtype',
                                                  'cont','criterion','LL'){1};
    dist.(variable_names{i}).fshift.min = min(squeeze(parameters(:,i+clen(5))));
    dist.(variable_names{i}).fshift.max = max(squeeze(parameters(:,i+clen(5))));
    dist.(variable_names{i}).covmat = cov(parameters(:,[i,i+num_bF,i+clen(5)]))
end


variable_names{end+1} = 'gaussian';
% variable_names{end+1} = 'gaussian_mm';
variable_names{end+1} = 'global_fshift';
variable_names{end+1} = 'phi0';
variable_names{end+1} = 'phi1';
variable_names{end+1} = 'snr';

names(strcmp(names,'ampl')) = [];
names(strcmp(names,'lorentzLB')) = [];
names(strcmp(names,'freqShift')) = [];
clen = clen([4,5,7,8,9]);
ind = ind{[3,4,6,7,8]};

for i=1:length(names)
    dist.(variable_names{i}) = fitmethis(parameters(:,ind{i}),'dtype','cont',
                                         'criterion','LL'){1};
    dist.(variable_names{i}).ampl.min = min(squeeze(parameters(:,ind{i})));
    dist.(variable_names{i}).ampl.max = max(squeeze(parameters(:,ind{i})));
end

save(fullfile(path,'distributions.mat'),'dist')

end
