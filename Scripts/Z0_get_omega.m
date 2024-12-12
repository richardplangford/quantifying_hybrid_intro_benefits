lastn = maxNumCompThreads(2);
clear all

dir_scripts = cd('..\Input');
load('bootstrap_mat.mat');

num_bootstraps = size(bootstrap_mat_group,2);

for cf_run=1:num_bootstraps
    
group = bootstrap_mat_group(cf_run);
run = bootstrap_mat_run(cf_run);
disp(['Group ',num2str(group),', Run ',num2str(run),':']);

fileloc = sprintf('full_moment_estimates_group%d_run%d.mat',group,run);
load(fileloc);

% Read in the proposed parameters
betaU = full_esttheta(1:numbetaU);
betaO = full_esttheta(numbetaU+1:numbetaU+numbetaXZ);
alpha = full_esttheta(numbetaU+numbetaXZ+1:numbetaU+numbetaXZ+numalpha);


% Calculate delta via the BLP contraction process
% norm, avgnorm, and tol determine the stopping criteria.
%
% If the contraction process produces an undefined delta at any point along
% the sequence, the code breaks, returns an undefined function value for
% the objection function, and resets delta to its initial point. Otherwise,
% the code continues to estimatation of the moments.
%
% mymktsh calculates individual and aggregate shares given the parameters.

mudraw = repmat(sum(bsxfun(@times,xz,permute(betaO,[2 3 1])),3),[1 draws]) +...
    sum(bsxfun(@times,xv,permute(betaU,[2 3 1])),3) +...
    repmat(sum(bsxfun(@times,priceinfo,permute(alpha,[2 3 1])),3),[1 draws]);

% Calculate individual shares, for each draw and in expectation, for the
% converged delta
[indsharedrawraw,~,aggsharedraw,~] = ...
        mymktsh(delta,mudraw,N,repmat(population,[1 draws]));
indsharedraw = sum(reshape(indsharedrawraw,[cdindex(T) N draws]),3)./draws;
    
incomebracketstemp = zeros(cdindex(T),N,numalpha);
for t=1:T
    incomebracketstemp(cdindexstart(t):cdindex(t),:,:) = repmat(permute(incomebrackets(:,N*(t-1)+1:N*t),[3 2 1]),[cdindex(t)-cdindexstart(t)+1 1 1]);
end
dmudrawprice = repmat(permute(...
        sum(...
            bsxfun(@times,...
                bsxfun(@times,permute(alpha,[3 2 1]),incomebracketstemp),...
            dpriceinfoprice),...
        3),...
    [3 2 1]),[1 draws 1]);
Delta = zeros(cdindex(T),cdindex(T));
for t=1:T
    temp = zeros(cdindex(t)-cdindexstart(t)+1,N,cdindex(t)-cdindexstart(t)+1);
    temp = bsxfun(@times,dmudrawprice(:,:,cdindexstart(t):cdindex(t)),bsxfun(@times,-indsharedrawraw(cdindexstart(t):cdindex(t),:),permute(indsharedrawraw(cdindexstart(t):cdindex(t),:),[3 2 1])));
    temp = temp + bsxfun(@times,permute(eye(cdindex(t)-cdindexstart(t)+1),[1 3 2]),bsxfun(@times,dmudrawprice(:,:,cdindexstart(t):cdindex(t)),permute(indsharedrawraw(cdindexstart(t):cdindex(t),:),[3 2 1])));
    Delta(cdindexstart(t):cdindex(t),cdindexstart(t):cdindex(t)) = -permute(bsxfun(@rdivide,sum(bsxfun(@times,repmat(population,[1 draws]),temp),2),sum(repmat(population,[1 draws]),2)),[1 3 2]);
end
Delta = Delta.*compete;
mchat = price - Delta\aggsharedraw;
clear temp Delta incomebracketstemp

omegahat = log(mchat.*10000) - C*estgamma;

fileloc = sprintf('full_moment_estimates_group%d_run%d.mat',group,run);
save(fileloc);











end



% mymktsh: calculates individual and aggregate shares
function [indsharedraw,outindsharedraw,aggsharedraw,outaggsharedraw] = mymktsh(delta,mu,N,population)

[indsharedraw,outindsharedraw] = myind_sh(delta,mu,N);
aggsharedraw = sum(bsxfun(@times,population,indsharedraw),2)./sum(population,2);
outaggsharedraw = sum(bsxfun(@times,population,outindsharedraw),2)./sum(population,2);

end

% myind_sh: calculates individual shares
function [indsharedraw,outindsharedraw] = myind_sh(delta,mu,N)

global cdid T draws cdindexstart cdindex

eg = exp(bsxfun(@plus,delta,mu));
sum1 = zeros(T,N*draws);

for t=1:T
    sum1(t,:) = sum(eg(cdindexstart(t):cdindex(t),:),1);
end

outindsharedraw = 1./(1+sum1);
denom = outindsharedraw(cdid,:);

indsharedraw = eg.*denom;
end