function [dist,parameters] = prepareDistributions(exportParams_path,vargin)
% This function will analyze the exported Osprey variables and several other
% possible variables such as B0 field map and coil-wise distributions of SNR,
% coil sensitivities, frequency drift, and phase drift.
% 
% exportParams_path is the path of the .mat file from Osprey.
% savedir can be used to specify a save directory
% 
% Required shapes:
%   - B0: [voxels, x_dir, y_dir, z_dir]
%   - coil vars: [voxels, coils]
%       - These should be in the following order:
%           - SNR, sensitivities, freq drift, phase drift
% 

if isstring(exportParams_path)
    path = exportParams_path; % /path/to/osp_exportParams.mat/file
    dist, parameters = analyzeOspreyParams(path);
elif isstruct(exportParams_path)
    dist = exportParams_path;
    for i=1:2:length(vargin)
        if istrcmp('parameters',vargin{i}):
            parameters = vargin{i+1};
fields = fieldnames(dist);
path = []

name = {'coil_snr','coil_sens','coil_fshift','coil_phi0'};
for i=1:2:length(vargin)
    if istrcmp('B0',vargin{i}):
        fields{end+1} = 'B0';
        fields{end+1} = 'B0_dir';
        data = vargin{i+1};
        dx_0 = data(:,1,1,1) - data(:,1,end,1);
        dx_1 = data(:,end,1,1) - data(:,end,end,1);
        dx_2 = data(:,1,1,end) - data(:,1,end,end);
        dx_3 = data(:,end,1,end) - data(:,end,end,end);
        dx = ((dx_0 + dx_1 + dx_2 + dx_3) ./ 4) ./ 2;

        dy_0 = data(:,1,1,1) - data(:,end,end,1);
        dy_1 = data(:,end,1,1) - data(:,1,end,1);
        dy_2 = data(:,1,1,end) - data(:,end,end,end);
        dy_3 = data(:,end,1,end) - data(:,1,end,end);
        dy = ((dy_0 + dy_1 + dy_2 + dy_3) ./ 4) ./ 2;

        dz_0 = data(:,1,1,1) - data(:,1,1,end);
        dz_1 = data(:,end,end,1) - data(:,end,end,end);
        dz_2 = data(:,1,end,1) - data(:,1,end,end);
        dz_3 = data(:,end,1,1) - data(:,end,1,end);
        dz = ((dz_0 + dz_1 + dz_2 + dz_3) ./ 4) ./ 2;

        mu = mean(data,[2,3,4]);

        parameters(:,end+1) = mu;
        parameters(:,end+1:end+3) = [dx, dy, dz];

        dist.B0.dx = dx;
        dist.B0.dy = dy;
        dist.B0.dz = dz;
        dist.B0.mean = mu;
        dist.B0.min = min(data,[],[1,2,3,4]);
        dist.B0.max = max(data,[],[1,2,3,4]);
        dist.B0.covmat = cov([dx, dy, dz, mu]);
        dist.B0.F = fitmethis(reshape(data.',1,[]),'dtype','cont',...
                              'criterion','LL'){1};%'
    else:
        if istrcmp('savedir',vargin{i}):
            path = vargin{i+1};
        else:
            for ii = 1:length(name)
                if istrcmp(name{ii},vargin{i}):
                    fields{end+1} = name{ii};
                    data = vargin{i+1};
                    sz = size(data);
                    dist.(name{ii}).min = min(data,[],[1,2]);
                    dist.(name{ii}).max = max(data,[],[1,2]);
                    dist.(name{ii}).mean = mean(data,'all');
                    dist.(name{ii}).covmat = cov(data);
                    dist.(name{ii}).F = fitmethis(reshape(data.',1,[]),'dtype',...
                                                  'cont','criterion','LL'){1}; %'
                    parameters(:,end+1:sz(end)) = data;
                end
            end
        end
    end

    if isstring(path)
        save(fullfile(path,'distributions.mat'),'dist','parameters')
    end

end
