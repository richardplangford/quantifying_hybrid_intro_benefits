function f = D1_full_objfuncgmm(theta)
% This function computes the GMM objective function.

global lastf lastgrad lasttheta delta tol cdid cdindex cdindexstart N draws T estbetamu estgamma ...
    numbetaXZ numbetaU numalpha betaXZindex betaUindex aggshare indshare price ...
    X IV Z W C xv xz priceinfo dpriceinfoprice incomebrackets compete ...
    population outshare Omega omegahat V var_cov var_cov_delta var_cov_beta

% Read in the proposed parameters
betaU = theta(1:numbetaU);
betaO = theta(numbetaU+1:numbetaU+numbetaXZ);
alpha = theta(numbetaU+numbetaXZ+1:numbetaU+numbetaXZ+numalpha);


% Calculate delta via the BLP contraction process
% norm, avgnorm, and tol determine the stopping criteria.
%
% If the contraction process produces an undefined delta at any point along
% the sequence, the code breaks, returns an undefined function value for
% the objection function, and resets delta to its initial point. Otherwise,
% the code continues to estimatation of the moments.
%
% mymktsh calculates individual and aggregate shares given the parameters.

i = 0;
norm = 1;
avgnorm = 1;
mudraw = repmat(sum(bsxfun(@times,xz,permute(betaO,[2 3 1])),3),[1 draws]) +...
    sum(bsxfun(@times,xv,permute(betaU,[2 3 1])),3) +...
    repmat(sum(bsxfun(@times,priceinfo,permute(alpha,[2 3 1])),3),[1 draws]);

while norm > tol && avgnorm > 1e-3*tol && i < 1000
    [~,~,aggsharedraw,~] = ...
        mymktsh(delta,mudraw,N,repmat(population,[1 draws]));
    new_delta = delta + log(aggshare) - log(aggsharedraw);
    if max(isnan(new_delta)) == 1
        break;
    end
    w = abs(new_delta - delta);
    norm = max(w);
    avgnorm = mean(w);
    delta = new_delta;
    i = i+1;
end
if max(isnan(new_delta)) == 1
    disp('Undefined delta; returning NaN...')
    delta = log(aggshare) - log(outshare);
	f = NaN;
else
% Calculate individual shares, for each draw and in expectation, for the
% converged delta
[indsharedrawraw,~,aggsharedraw,~] = ...
        mymktsh(delta,mudraw,N,repmat(population,[1 draws]));
indsharedraw = sum(reshape(indsharedrawraw,[cdindex(T) N draws]),3)./draws;
    
% Use 2SLS to calculate betamu, the coefficients on the non-price vehicle
% attributes

P = IV/(IV'*IV)*IV';
estbetamu = (X'*P*X)\(X'*P*delta);
xihat = delta - X*estbetamu;

% Calculate G1, the macro BLP demand moments
G1temp = bsxfun(@times,IV,xihat);
G1 = mean(G1temp,1)';

% Calculate G2 and G3, the micro BLP demand moments, matching expected and
% observed covariance in vehicle and consumer attributes

%%% G2
actual = bsxfun(@rdivide,...
    sum(bsxfun(@times,xz,bsxfun(@times,population,indshare)),2),...
    sum(bsxfun(@times,population,indshare),2));
expected = bsxfun(@rdivide,...
    sum(bsxfun(@times,xz,bsxfun(@times,population,indsharedraw)),2),...
    sum(bsxfun(@times,population,aggshare),2));
temp = actual-expected;
popweight = sum(bsxfun(@times,population,indshare),2)./sum(sum(bsxfun(@times,population,indshare),2),1);
G2temp =  permute(temp,[1 3 2]);
% If a vehicle has no purchases in a given year, we exclude that instance
% from the calculation of G2
% G2 = nanmean(G2temp,1)';
G2 = sum(bsxfun(@times,popweight,G2temp),1)';
% G2temp(isnan(G2temp)) = 0;

%%% G3
actual = bsxfun(@rdivide,...
    sum(bsxfun(@times,priceinfo,bsxfun(@times,population,indshare)),2),...
    sum(bsxfun(@times,population,indshare),2));
expected = bsxfun(@rdivide,...
    sum(bsxfun(@times,priceinfo,bsxfun(@times,population,indsharedraw)),2),...
    sum(bsxfun(@times,population,aggshare),2));
temp = actual-expected;
popweight = sum(bsxfun(@times,population,indshare),2)./sum(sum(bsxfun(@times,population,indshare),2),1);
G3temp =  permute(temp,[1 3 2]);
% If a vehicle has no purchases in a given year, we exclude that instance
% from the calculation of G3
% G3 = nanmean(G3temp,1)';
G3 = sum(bsxfun(@times,popweight,G3temp),1)';
% G3temp(isnan(G3temp)) = 0;

% Calculate the macro BLP supply moments
% First get the Delta matrix, made up of derivatives of shares with respect to price for
% competing vehicle prices and 0 otherwise.
incomebracketstemp = zeros(cdindex(T),N,numalpha);
for t=1:T
    incomebracketstemp(cdindexstart(t):cdindex(t),:,:) = repmat(permute(incomebrackets(:,N*(t-1)+1:N*t),[3 2 1]),[cdindex(t)-cdindexstart(t)+1 1 1]);
end
dbetamuprice = estbetamu(14);
dmudrawprice = permute(...
        sum(...
            bsxfun(@times,...
                bsxfun(@times,permute(alpha,[3 2 1]),incomebracketstemp),...
            dpriceinfoprice),...
        3),...
    [3 2 1]);
dmudrawprice = bsxfun(@plus,dbetamuprice,dmudrawprice);
Delta = zeros(cdindex(T),cdindex(T));
for t=1:T
    temp = zeros(cdindex(t)-cdindexstart(t)+1,N,cdindex(t)-cdindexstart(t)+1);
    temp = bsxfun(@times,dmudrawprice(:,:,cdindexstart(t):cdindex(t)),bsxfun(@times,-indsharedraw(cdindexstart(t):cdindex(t),:),permute(indsharedraw(cdindexstart(t):cdindex(t),:),[3 2 1])));
    temp = temp + bsxfun(@times,permute(eye(cdindex(t)-cdindexstart(t)+1),[1 3 2]),bsxfun(@times,dmudrawprice(:,:,cdindexstart(t):cdindex(t)),permute(indsharedraw(cdindexstart(t):cdindex(t),:),[3 2 1])));
    Delta(cdindexstart(t):cdindex(t),cdindexstart(t):cdindex(t)) = -permute(bsxfun(@rdivide,sum(bsxfun(@times,population,temp),2),sum(population,2)),[1 3 2]);
end
Delta = Delta.*compete;
mchat = price - Delta\aggsharedraw;
clear temp

% Use 2SLS to solve for gamma
P = IV/(IV'*IV)*IV';
estgamma = (C'*P*C)\(C'*P*log(mchat.*10000));
omegahat = log(mchat.*10000) - C*estgamma;

% Calculate G4, the macro BLP demand moments
G4temp = bsxfun(@times,IV,omegahat);
G4 = mean(G4temp,1)';

% If the estimated marginal costs are less than zero, the function returns
% undefined values and restarts.
% if min(mchat)<0 
%     disp('Negative marginal costs...')
%     norm = min(mchat);
%     delta = log(aggshare) - log(outshare);
%     G = [G1;G2;G3;G4];
%     f = real(G'/W*G - norm*1000);
% else

% Concatenate the moments and return the objective function.

G = [G1;G2;G3;G4];
f = real(G'/W*G);

% end

end

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