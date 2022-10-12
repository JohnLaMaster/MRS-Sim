function F = fitmethis(data,varargin)
% FITMETHIS finds best-fitting distribution
%	F = fitmethis(X) fits all distributions available in MATLAB's function 
%	MLE to data in vector X and returns them ordered by either LogLikelihood 
%	or Akaike. Either 21 continuous or 5 discrete distributions
%	are used based on user input or the type of data supplied (see below).
%	A non-parametric fit is also included for 'difficult' data.
%
%	The function returns a structure array with fields: 
%	name: name of the distribution (see HELP MLE for a list)
%	par:  vector of parameter estimates (1, 2 or 3 values)
%	ci:   matrix of confidence limits, one column per parameter
%	LL:   Log-Likelihood of the data
%	aic:  Akaike Information Criterion
% 
% 	Input data should be a vector or a 2D matrix in the shape [samples, 
% 	observations]. In the case of a 2D matrix, a covariance matrix 'cov' 
% 	will be included in the output. For a 1D matrix, 'cov' = -1.
%
%	Optional arguments can be specified as name/value pairs. Argument 
%	names are case insensitive and partial matches are allowed.
%
%	Name		Value
%	'dtype'		character string that specifies if the data are continuous 
% 				('cont') or discrete ('disc'). If missing, the function 
% 				decides the data are discrete if all values of X are 
% 				natural numbers.
%
%	'ntrials' 	Specifies number of trials for the binomial distribution. 
% 				NTRIALS must be either a scalar or a vector of the same 
% 				size as X. If missing the binomial is not fitted.
%
%	'figure'	Either 'on' (default), or 'off'. If 'on' a plot of the 
% 				data and the best fitting distribution is produced (scaled 
%				to match the data). Requires aditional function 'plotfitdist'.
% 
%	'pdata'		Either 'on' (default) or 'off'. If 'on' a histogram of the
% 				data will be plotted.
%
%	'pdist'		Number of distributions to plot, starting with the best-fitting.
%				Default is 1 and maximum is 4. If a preferred distribution is chosen, 
%				only this will	be plotted. 
% 
%	'alpha'		A value between 0 and 1 specifying a confidence level
%				for CI of 100*(1-alpha) (default is 0.05).
%
%	'criterion'	Criterion used to order the fits. It can be: 'LL' for
%				Log-Likelihood (default), or 'AIC' for Akaike.
% 
%	'output'	If set to 'off' supresses output to the command window.
%				Default 'on'
%
%	'pref'		Name of the preferred distribution to plot
% 
%	'kernel'	Include a non-parametric. Default: false.
%
%	If X contains negative values, only the Normal distribution is fitted.
%	Also, if X contains values > 1 the Beta distribution is not fitted. If X
%	contains 0 some distributions are not fitted.
%
%	Requires Statistics Toolbox
% 
%	Example 1:
%	x = gamrnd(5,0.5,1000,1);
%	F = fitmethis(x);
%
%    Name      Par1       Par2        Par3         LogL
%   gamma   4.947e+00   5.034e-01               -1.461e+03
%     gev  -1.126e-02   8.965e-01   1.982e+00   -1.463e+03
%	  ...
% 
%	Example 2 (small bump to highlight kernel):
%	x = gamrnd(5,0.5,1000,1);
%	y = normrnd(5,0.3,80,1);
%	F = fitmethis([x; y],'pdist',4)
% 
% 
% 	Modified by Ing. John T LaMaster 10.06.2022


warning off


% Covariance matrix
if ~isvector(data) and ismatrix(data)
	data = reshape(data.',1,[]); %''
end


% Defaults & Storage
dtype = {};
ntrials = [];
fig = 'on';
alpha = 0.05;
criterion = 'LL';
output = 'on';
prefdist = [];
plotdata = 'on';
plotdist = 1;
applykernel = false;


% Arguments
for j = 1:2:length(varargin)
	string = lower(varargin{j});
	switch string(1:min(4,length(string)))
		case 'dtyp'
			dtype = varargin{j+1};
		case 'ntri'
			ntrials = varargin{j+1};
		case 'figu'
			fig = varargin{j+1};
		case 'alph'
			alpha = varargin{j+1};
		case 'crit'
			criterion = varargin{j+1};
		case 'outp'
			output = varargin{j+1};
		case 'pref'
			prefdist = varargin{j+1};
		case 'pdat'
			plotdata = varargin{j+1};
		case 'pdis'
			plotdist = varargin{j+1};
		case 'kern'
			applykernel = varargin{j+1};
		otherwise
			error('Unknown argument name');
	end
end



f_data = struct('min',data_mn,'max',data_mx,'mu',data_mu,...
		   		'num',numel(data),'cov',covmat);
F = struct('name',{},'par',[],'ci',[],'LL',[],'aic',[],'data',{});


% Distributions
Cdist = {'beta'; 'birnbaumsaunders'; 'burr'; 'chi'; 'exponential'; 'ev'; ...
 		 'gamma'; 'gev'; 'gp'; 'hn'; 'inversegaussian'; 'logistic'; ...
	 	 'loglogistic'; 'lognormal'; 'nakagami'; 'normal'; 'rayleigh'; ...
		 'rician'; 'tlocationscale'; 'uniform'; 'weibull'};
mustbepos = 11;

Ddist = {'binomial'; 'geometric'; 'nbin'; 'unid'; 'poisson'};



% Try determine data type: Discrete or Continuous (avoid 0)
if isempty(dtype)
	if isempty(find(1 - (data+1)./(fix(data)+1), 1)) 
		dtype = 'disc';
	else
		dtype = 'cont';
	end
end



% Fit all... 
switch dtype(1:4)

	% Continuous
	case 'cont'
	for j = 1:numel(Cdist)
		
		% If negative values, only fit normal
		if min(data) < 0
			[phat,pci]= mle(data,'distribution','normal','alpha',alpha);
			F(j).name = Cdist{j};
			F(j).par = phat;
			F(j).ci = pci;
			pdfv = pdf('normal',data,F(j).par(1),F(j).par(2));
			F(j).LL = sum(log(pdfv(pdfv>0 & ~isinf(pdfv))));
			F(j).aic = 2*2 - 2*F(j).LL;
			F(j).data = f_data;
			break

		% Check: if values > 1 for Beta, do nothing
		elseif strcmp('beta',Cdist{j}) && max(data) > 1
			F(j).name = 'beta';
			F(j).LL = -Inf;
			F(j).aic = Inf;
			F(j).data = f_data;

		% Check: if values > 0 for some distr. (they are sorted), do nothing
		elseif  j >= mustbepos && min(data) == 0
			F(j).name = Cdist{j};
			F(j).LL = -Inf;
			F(j).aic = Inf;
			F(j).data = f_data;

		% Any other case do the fit ...
		else
			try
				[phat,pci] = mle(data,'distribution',Cdist{j},'alpha',alpha);
				F(j).name = Cdist{j};
				F(j).par = phat;
				F(j).ci = pci;
				if numel(F(j).par) == 1
					pdfv= pdf(F(j).name,data,F(j).par(1));
				elseif numel(F(j).par) == 2
					pdfv = pdf(F(j).name,data,F(j).par(1),F(j).par(2));
				else
					pdfv = pdf(F(j).name,data,F(j).par(1),F(j).par(2),F(j).par(3));
				end
				F(j).LL = sum(log(pdfv(pdfv>0 & ~isinf(pdfv))));
				F(j).aic = 2*numel(F(j).par) - 2*F(j).LL;
				F(j).data = f_data;
			catch
				F(j).name = Cdist{j};
				F(j).par = NaN;
				F(j).LL = -Inf;
				F(j).aic = Inf;
				F(j).data = f_data;
				fprintf('%s%s\n',Cdist{j},' distribution not applicable')
			end
		end

	end


	% Add non-parametric (kernel) estimation with ksdensity. The degrees 
	% of freedom (required for AIC) can be estimated as the trace of the 
	% smoothing matrix. This matrix is not returned by ksdensity, so I 
	% create it here following descriptions of kernel estimation 
	% assuming a normal distribution. The pdf estimates that this procedure 
	% returns (Kpdf2) are identical to those of ksdensity. They are not 
	% required for anything, calculated only as a test.
	% In general, kernel estimators will have higher LogLikelihood but, due
	% to the high number of parameters, they will have much higher AICs
	% (undesirable).
	% This is not applied by default, to avoid out of memory problems for
	% large data sets.
	if applykernel
		warning('The kernel option may cause memory problems for large data sets')
		[Kpdf,mesh,bandw] = ksdensity(data,data);
		F(j+1).name = 'Kernel';
		F(j+1).par = bandw;
		F(j+1).LL = sum(log(Kpdf));

		N = numel(data);
		w = zeros(N);
		S = zeros(N);
		for k = 1:N
			S(k,:) = normalf(data,data(k),bandw);
			w(k,:) = normalf(mesh,data(k),bandw);
		end
		Kpdf2 = sum(w)/N; %#ok<*NASGU>
		dof = trace(S);
		F(j+1).aic = 2*dof - 2*F(j+1).LL;
		F(j+1).data = f_data;
	end


	% Discrete
	case 'disc'
	for j = 1:numel(Ddist)

		% Binomial needs number of trials
		if strcmp('binomial',Ddist{j}) 
			F(j).name = 'binomial';
			if isempty(ntrials) || (numel(ntrials) > 1 && numel(data) ~= numel(ntrials))
				F(j).LL = -Inf;
				F(j).aic = Inf;
				F(j).data = f_data;
			else
				[phat,pci] = mle(data,'ntrials',ntrials,'distribution','binomial','alpha',alpha);
				F(j).par = phat;
				F(j).ci = pci;
				pdfv = pdf('bino',data,ntrials,F(j).par(1));
				F(j).LL = sum(log(pdfv(pdfv>0 & ~isinf(pdfv))));
				F(j).ntrials = ntrials;
				F(j).data = f_data;
			end
		else
			[phat,pci] = mle(data,'distribution',Ddist{j},'alpha',alpha);
			F(j).name = Ddist{j};
			F(j).par = phat;
			F(j).ci = pci;
			if numel(F(j).par) == 1
				pdfv = pdf(F(j).name,data,F(j).par(1));
			elseif numel(F(j).par) == 2
				pdfv = pdf(F(j).name,data,F(j).par(1),F(j).par(2));
			else
				pdfv = pdf(F(j).name,data,F(j).par(1),F(j).par(2),F(j).par(3));
			end
			F(j).LL = sum(log(pdfv(pdfv>0 & ~isinf(pdfv))));
			F(j).aic = 2*numel(F(j).par) - 2*F(j).LL;
			F(j).data = f_data;
		end
	end

end


% Order by criterion
switch criterion
	case 'LL'
		index = sortrows([(1:size(F,2))',[F.LL]'],-2);
	case 'AIC'
		index = sortrows([(1:size(F,2))',[F.aic]'],2);
end
F = F(index(:,1));


% Nice screen output
if strcmp('on',output)
	fprintf('\n\t\t\t\tName\t\tPar1\t\tPar2\t\tPar3\t\tLogL\t\tAIC\n')
	for j= 1:size(F,2)
		switch numel(F(j).par)
			case 1
				fprintf('%20s \t%10.3e \t\t\t\t\t\t\t%10.3e \t%10.3e\n',F(j).name,F(j).par,F(j).LL,F(j).aic)
			case 2
				fprintf('%20s \t%10.3e \t%10.3e \t\t\t\t%10.3e \t%10.3e\n',F(j).name,F(j).par,F(j).LL,F(j).aic)
			case 3
				fprintf('%20s \t%10.3e \t%10.3e \t%10.3e \t%10.3e \t%10.3e\n',F(j).name,F(j).par,F(j).LL,F(j).aic)
		end
	end
end


% Choose the preferred distr. for plotting (if specified)
if ~isempty(prefdist)
	best= find(strcmp(cellstr(prefdist),cellstr({F.name})));
	if isempty(best)
		if strcmp('on',output) 
			fprintf('\n *** Warning: the preferred distr. is not in the list of fitted \n');
		end
	else
		plotfitdist(data,F(best),dtype,'pdat',plotdata);
		if strcmp('on',output)
			fprintf('\n *** Warning: Plotting preferred distr. [%s] NOT BEST FIT \n',F(best).name);
		end
	end
end


% Plot data histogram & best fits from 1 to plotdist
if strcmp('on',fig) && isempty(prefdist)
	plotfitdist(data,F,dtype,'pdist',plotdist,'pdat',plotdata);
end

% End of fitmethis
end



function y = normalf(x,mu,sigma)
% Normal function values at x with mean= mu and std= sigma
y = (1/(sigma*sqrt(2*pi))) * exp(-((x-mu).^2)./(2*sigma^2));
end
