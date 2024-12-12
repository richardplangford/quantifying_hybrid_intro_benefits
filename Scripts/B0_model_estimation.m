% B0_model_estimation: sets up and runs the objective function solver to obtain estimates
lastn = maxNumCompThreads(2);
clear all


% global variables which will be sent to the objective function, prevents
%   function from having to load data each time it is called
global lastf lastgrad lasttheta delta tol cdid cdindex cdindexstart N draws T ...
    numbetaXZ numbetaU numalpha betaXZindex aggshare outshare indshare price ...
    X IV Z W C xv xz priceinfo dpriceinfoprice estbetamu estgamma omegahat ...
    compete population incomebrackets outshare Omega var_cov var_cov_delta var_cov_beta

% load prepared data
dir_scripts = cd('..\Log');
diary('demand_model_run.log')
diary on

dir_log = cd('..\Output');
load('prepped_data.mat');

% Set up vector of estimates

% betaU: variance of random coefficients
%   [2,1] = 10./mpg * N(0,1) draws
%   [3,2] = liters./10 * N(0,1) draws
%   [10,3] = hybrid * N(0,1) draws
%   [14,4] = price * N(0,1) draws
numbetaU = 4;
betaUindex = [2,1;3,2;10,3;14,4];

% betaXZ: combinations of vehicle and consumer attributes
%   [1,1] through [1,6] = age65, age18, unemprate, popdens, gasprice,
%   income
%   [10,6] = hybrid * income
%   [10,5] = hybrid * gasprice

numbetaXZ = 8;
betaXZindex = [1,1;1,2;1,3;1,4;1,5;1,6;10,5;10,6];

% alpha: price for 3 interactions
numalpha = 3;

% Set RNG to ensure replicability
rng(20200617);

% choose random starting values over a uniform distribution, keep bounds
%   very short to ensure initial point is defined
betaUstart = unifrnd(0,.001,numbetaU,1);
betaXZstart = unifrnd(-.001,.001,numbetaXZ,1);
alphastart = unifrnd(-.001,.001,numalpha,1);
starttheta = [betaUstart;betaXZstart;alphastart];
% lasttheta: keeps track of the previous estimate, used to speed up
%   estimation
lasttheta = zeros(size(starttheta));

% Draw from standard normal distribution to estimate rand. coefs.
randnu = normrnd(0,1,[N*draws*T numbetaU])';

% xv: matrix of unobserved vehicle attribute tastes
xv = zeros(cdindex(T),N*draws,numbetaU);
for i=1:numbetaU
    xvbig = repmat(X(:,betaUindex(i,1)),[1 N*draws*T]).*repmat(randnu(betaUindex(i,2),:),[cdindex(T) 1]);
    for t=1:T
        xv(cdindexstart(t):cdindex(t),:,i) = xvbig(cdindexstart(t):cdindex(t),N*draws*(t-1)+1:N*draws*t);
    end
end

% xz: matrix of vehicle and consumer attribute interactions
xz = zeros(cdindex(T),N,numbetaXZ);
for i=1:numbetaXZ
    xzbig = repmat(X(:,betaXZindex(i,1)),[1 N*T]).*repmat(Z(betaXZindex(i,2),:),[cdindex(T) 1]);
    for t=1:T
        xz(cdindexstart(t):cdindex(t),:,i) = xzbig(cdindexstart(t):cdindex(t),N*(t-1)+1:N*t);
    end
end

% priceinfo: matrix of price and consumer attribute interactions
% dpriceinfoprice: derivative of prices (not priceinfo right now)
% incomebrackets: 1 if consumer fits into income quartile, 0 otherwise
% Lower-middle-40%, Higher-middle-40%, Top 10%
priceinfo = zeros(cdindex(T),N,numalpha);
incomebrackets = [(income>3.6741 & income<=5.7991);...
    (income>5.7991 & income<=9.5547);...
    (income>9.5547)];
c = 0;
for t=1:T
    priceinfotemp = repmat(price(cdindexstart(t):cdindex(t),:),[1 N numalpha]);
    dpriceinfotempprice = repmat(ones(size(price(cdindexstart(t):cdindex(t),:))),[1 N]);
    priceinfo(cdindexstart(t):cdindex(t),:,:) = priceinfotemp.*repmat(permute(incomebrackets(:,N*(t-1)+1:N*t),[3 2 1]),[cdindex(t)-cdindexstart(t)+1 1 1]);
    dpriceinfoprice(cdindexstart(t):cdindex(t),:) = dpriceinfotempprice;
end

% W: weighting matrix for the GMM estimator
W = eye(size(IV,2)+numbetaXZ+numalpha);
W(1:size(IV,2),1:size(IV,2)) = IV'*IV;

% set the tolerance for the inner fixed-point loop
tol = 1e-13;

% Set options for constrained objective function minimizer
options = optimoptions('fmincon','GradObj','on','TolFun',1e-6,'Display','iter-detailed','FinDiffType','central',...
    'MaxFunEvals',1e+10,'SubproblemAlgorithm','cg','MaxIterations',10000,...
    'SpecifyObjectiveGradient',true,'CheckGradients',false,'Diagnostics','on');
cd(dir_scripts);

% Remove all vehicles with price above $90K from the market; adjust data accordingly
price_orig = price;
delta_orig = delta;
Xreal = X;
Xtrans = X';
xv = xv(X(:,18)==0,:,:);
xz = xz(X(:,18)==0,:,:);
priceinfo = priceinfo(X(:,18)==0,:,:);
dpriceinfoprice = dpriceinfoprice(X(:,18)==0,:);
price = price(X(:,18)==0,:);
IV = IV(X(:,18)==0,:);
aggshare = aggshare(X(:,18)==0,:);
indshare = indshare(X(:,18)==0,:);
compete = compete(X(:,18)==0,Xtrans(18,:)==0);
C = C(X(:,18)==0,:);
full_manu = full_manu(X(:,18)==0,:);


X = X(X(:,18)==0,:);
X = [X(:,1:14),X(:,19)];

realcdindex = cdindex;
realcdindexstart = cdindexstart;
cdid = X(:,11);
cdidshift = [1;cdid];
temp = [cdid;9] - cdidshift;
cdindex = find(temp) - 1;
cdindexstart = [0;cdindex(1:7)] + 1;
clearvars cdidshift temp

% calculate initial estimate of delta
temp = aggshare;
sum1 = zeros(T,1);
for t=1:T
    sum1(t) = sum(temp(cdindexstart(t):cdindex(t)),1);
end
outshare = 1.0 - sum1(cdid,:);
delta = log(aggshare./outshare);
clear temp sum1

% Estimate the model
% Bounds set to improve estimation time of betaU
esttheta = fmincon(@B1_objfuncgmm,starttheta,[],[],[],[],...
    [zeros(numbetaU,1);ones(numbetaXZ+numalpha,1).*-Inf],...
    [ones(numbetaU,1)*10;ones(numbetaXZ+numalpha,1).*Inf],[],options);
esttheta_firststep = esttheta;
estbetamu_firststep = estbetamu;
var_cov_firststep = var_cov;
var_cov_delta_firststep = var_cov_delta;
var_cov_beta_firststep = var_cov_beta;
std_err_firststep = sqrt(diag(var_cov./cdindex(T)));
std_err_beta_firststep = sqrt(diag(var_cov_beta./cdindex(T)));

% Save output
dir_scripts = cd('..\Output');
save 'demand_moment_estimates.mat'

% cd(dir_scripts);
% W = Omega;
% newstarttheta = esttheta + unifrnd(-.0001,.0001,size(starttheta,1),1);
% newesttheta = fmincon(@B1_objfuncgmm,starttheta,[],[],[],[],...
%     [zeros(numbetaU,1);ones(numbetaXZ+numalpha,1).*-Inf],...
%     [ones(numbetaU,1)*10;ones(numbetaXZ+numalpha,1).*Inf],[],options);
% esttheta_secondstep = newesttheta;
% estbetamu_secondstep = estbetamu;
% var_cov_secondstep = var_cov;
% var_cov_delta_secondstep = var_cov_delta;
% var_cov_beta_secondstep = var_cov_beta;
% std_err_secondstep = sqrt(diag(var_cov./cdindex(T)));
% std_err_beta_secondstep = sqrt(diag(var_cov_beta./cdindex(T)));
% dir_scripts = cd('..\Output');
% save 'demand_moment_estimates_twostep.mat'










load 'demand_moment_estimates.mat'
% 
% % Return current folder to scripts
% cd(dir_scripts);
% 
% % W: weighting matrix for the GMM estimator
% W = eye(size(IV,2)+numbetaXZ+numalpha);
% W(1:size(IV,2),1:size(IV,2)) = IV'*IV;
% W(size(IV,2)+numbetaXZ+numalpha+1:size(IV,2)+numbetaXZ+numalpha+size(IV,2),...
%     size(IV,2)+numbetaXZ+numalpha+1:size(IV,2)+numbetaXZ+numalpha+size(IV,2)) = IV'*IV;
% 
% % set the tolerance for the inner fixed-point loop
% tol = 1e-13;
% options = optimset('GradObj','off','TolFun',1e-6,'Display','iter-detailed','FinDiffType','central',...
%     'MaxFunEvals',1e+10,'SubproblemAlgorithm','cg');
% % esttheta = fmincon(@B1_objfuncgmm,starttheta,[],[],[],[],...
% %     [0;ones(numbetaXZ+4,1).*-Inf],...
% %     ones(numbetaU+numbetaXZ+4,1).*Inf,[],options);
% full_esttheta = fmincon(@D1_full_objfuncgmm,esttheta,[],[],[],[],...
%     [zeros(numbetaU,1);ones(numbetaXZ+numalpha,1).*-Inf],...
%     [ones(numbetaU,1)*10;ones(numbetaXZ+numalpha,1).*Inf],[],options);
% 
% % Save output
% dir_scripts = cd('..\Output');
% save 'full_moment_estimates.mat'

% Return current folder to scripts
cd(dir_scripts);
diary off
