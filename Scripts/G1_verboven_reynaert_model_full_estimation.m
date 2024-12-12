% B0_model_estimation: sets up and runs the objective function solver to obtain estimates
lastn = maxNumCompThreads(2);
clear all


% global variables which will be sent to the objective function, prevents
%   function from having to load data each time it is called
global lastf lastgrad lasttheta delta tol cdid cdindex cdindexstart N draws T ...
    numbetaXZ numbetaU numalpha betaXZindex aggshare outshare indshare price ...
    X IV Z W C xv xz priceinfo dpriceinfoprice estbetamu estgamma omegahat ...
    compete population incomebrackets outshare Omega var_cov var_cov_delta var_cov_beta ddeltatheta

% load prepared data
dir_scripts = cd('..\Log');
diary('verboven_reynaert_full_run.log')
diary on

dir_log = cd('..\Output');
load('demand_moment_estimates.mat');
lasttheta = zeros(size(esttheta));

dir_output = cd('..\Scripts');
B1_objfuncgmm(esttheta);
IV = [C,ddeltatheta];

% W: weighting matrix for the GMM estimator
W = eye(size(IV,2)+numbetaXZ+numalpha);
W(1:size(IV,2),1:size(IV,2)) = IV'*IV;
W(size(IV,2)+numbetaXZ+numalpha+1:size(IV,2)+numbetaXZ+numalpha+size(IV,2),...
    size(IV,2)+numbetaXZ+numalpha+1:size(IV,2)+numbetaXZ+numalpha+size(IV,2)) = IV'*IV;

% set the tolerance for the inner fixed-point loop
tol = 1e-13;

% Set options for constrained objective function minimizer
options = optimset('GradObj','off','TolFun',1e-6,'Display','iter-detailed','FinDiffType','central',...
    'MaxFunEvals',1e+10,'SubproblemAlgorithm','cg');
% Estimate the model
%   Bounds set to improve estimation time of betaU
verboven_reynaert_full_esttheta = fmincon(@D1_full_objfuncgmm,starttheta,[],[],[],[],...
    [zeros(numbetaU,1);ones(numbetaXZ+numalpha,1).*-Inf],...
    [ones(numbetaU,1)*10;ones(numbetaXZ+numalpha,1).*Inf],[],options);

% Save output
dir_scripts = cd('..\Output');
save 'verboven_reynaert_full_estimates.mat'

% Return current folder to scripts
cd(dir_scripts);
diary off
