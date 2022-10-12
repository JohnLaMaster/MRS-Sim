function plotfitdist(data,F,dtype,varargin)
% PLOTFITDIST plots data and fitted distribution
% PLOTFITDIST(DATA,F,DTYPE) plots one or more probability density 
% functions (specified in structure F), and the histogram 
% of values in vector DATA scaled by total area using 'trapz'. 
% 
% Structure array F must contain the following fields:
%  'name' (char. string) is the name of the distribution
%	(for examples of accepted names see 'HELP PDF').
%  'par' a vector of parameters of the distribution. The size of 'par'
%  (1 to 3) must match the requirements of function PDF for 
%  the specified distribution.
%  'F' may contain more than 1 distribution specification. Up to 4
%  distributions will be plotted, as specified in optional argument 'pdist'.
%  
%  DTYPE is a character string, either 'cont' for continuous 
%	distributions or 'disc' for discrete ones. Continuous 
%	distributions are plotted as a line, while discrete ones are 
%	plotted as an histogram superimposed on the data histogram.
% 
% If 'name' is 'binomial', structure F must have an additional 
%  field called 'ntrials', which must be a scalar specifying the number 
%  of trials. If 'ntrials' is a vector (i.e. different number of 
%  trials for each value in DATA) the plot cannot be made due to 
%  mismatch in the number of simulated data and number of trials.
% 
% Optional arguments:
% 'lwidth' Width of plotted line (default 1.5).
% 'legend' Wether to plot a legend. 'on' (default) or 'off'. The legend 
%          will be the names of each distribution plotted.
% 'pdata'  Wether to plot the data. 'on' (default), or 'off'.
% 'pdist'  Number of distributions to plot (1 to 4). Default 1. If the 
%          size F is smaller than pdist, it will be ignored.
% 
% The function is designed to work with 'fitmethis'. However, it can be
% used independently if the input arguments are provided as specified.


% Defaults
linewidth = 1.5;
legnd     = 'on';
pdat      = 'on';
pdist     = 1;
linecol   = [.6 .8 1; 1 .6 0; .6 .8 0; .8 .6 1];


% Arguments
for j = 1:2:length(varargin)
	string = lower(varargin{j});
	switch string(1:4)
		case 'lwid'
			linewidth= varargin{j+1};
		case 'lege'
			legnd= varargin{j+1};
		case 'pdat'
			pdat= varargin{j+1};
		case 'pdis'
			pdist= varargin{j+1};
		otherwise
			error(['Unknown argument name: ',varargin{j}]);
	end
end


% Histogram for Discrete/Continuous
if strcmp(dtype,'cont')
	x = min(data):range(data)/50:max(data);
	[bincount,binpos] = hist(data,min(50,numel(data)/5));
else
	x = unique(data);
	[bincount,binpos] = hist(data,x);
end


% Plot 1 or more distributions
figure(gcf); 
hold on


% Plot data first (just once)
bincount = bincount/trapz(binpos,bincount); % scaled frequencies
if strcmp(pdat,'on')
	datap = bar(binpos,bincount,'FaceColor',[.85 .85 .85],'EdgeColor',[1 1 1],'BarWidth',1);
end
set(gca,'Layer','top'); % Avoids lower edge of bars covering the axis. 

% Plot 'pdist' best distributions (up to 4 or size of F)
n2plot = min([pdist,4,size({F.name},2)]);
for j = 1:n2plot

	distname = F(j).name;
	par =      F(j).par;

	% Extract ntrials if it is there
	if isfield(F,'ntrials')
		ntrials = F(j).ntrials;
	end

	% Calculate predicted
	switch numel(par)
		case 1
			if strcmp('binomial',distname)
				y = pdf('bino',x,ntrials(1),par(1));
			elseif strcmp('Kernel',distname)
				y = ksdensity(data,x,'Bandwidth',par(1));
			else
				y = pdf(distname,x,par(1));
			end
		case 2
			y = pdf(distname,x,par(1),par(2));
		case 3
			y = pdf(distname,x,par(1),par(2),par(3));
	end

	% Plot distrib. pdf
	if strcmp('cont',dtype)
		model = plot(x,y,'r','LineWidth',linewidth,'Color',linecol(j,:));
	else
		model = bar(x,y,'FaceColor',linecol(j,:),'EdgeColor','none','BarWidth',0.5);
	end

end

xlabel('Data'); ylabel('PDF');

% Plot legend with the appropriate number of distr. names
if strcmp(legnd,'on')
	legend(['data' {F(1:n2plot).name}]); legend('boxoff'); 
end
