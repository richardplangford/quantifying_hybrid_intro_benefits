function [f,grad] = B1_objfuncgmm(theta)
% This function computes the GMM objective function.

global lastf lastgrad lasttheta delta tol cdid cdindex cdindexstart N draws T estbetamu ...
    numbetaXZ numbetaU numalpha aggshare indshare ...
    X IV W xv xz priceinfo ...
    population outshare V ddeltatheta
tic
% if theta==lasttheta
%     f = lastf;
%     grad = lastgrad;
% else

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
    grad = theta.*NaN;
else
% Calculate individual shares, for each draw and in expectation, for the
% converged delta
[indsharedrawraw,~,~,~] = ...
        mymktsh(delta,mudraw,N,repmat(population,[1 draws]));
indsharedraw = sum(reshape(indsharedrawraw,[cdindex(T) N draws]),3)./draws;
    
% Use 2SLS to calculate betamu, the coefficients on the non-price vehicle
% attributes

temp1 = IV'*X;
temp2 = IV'*delta;
tempW = IV'*IV;
estbetamu = (temp1'/tempW*temp1)\(temp1'/tempW*temp2);
clear temp1 temp2 tempinvW

% Calculate residuals to form G1, the macro BLP demand moments
xihat = delta - X*estbetamu;
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
G2temp =  permute(temp,[1 3 2]);
% If a vehicle has no purchases in a given year, we exclude that instance
% from the calculation of G2
G2 = nanmean(G2temp,1)';
G2temp(isnan(G2temp)) = 0;

%%% G3
actual = bsxfun(@rdivide,...
    sum(bsxfun(@times,priceinfo,bsxfun(@times,population,indshare)),2),...
    sum(bsxfun(@times,population,indshare),2));
expected = bsxfun(@rdivide,...
    sum(bsxfun(@times,priceinfo,bsxfun(@times,population,indsharedraw)),2),...
    sum(bsxfun(@times,population,aggshare),2));
temp = actual-expected;
G3temp =  permute(temp,[1 3 2]);
% If a vehicle has no purchases in a given year, we exclude that instance
% from the calculation of G3
G3 = nanmean(G3temp,1)';
G3temp(isnan(G3temp)) = 0;


%%%%% MOMENTS HAVE BEEN CALCULATED %%%%%

%%%%% GRADIENT CALCULATION %%%%%

% Calculate (partial indsharedrawraw)/(partial betaU)...
temp = bsxfun(@times,xv,indsharedrawraw);
sum1 = zeros(T,N*draws,numbetaU);
for t=1:T
    sum1(t,:,:) = sum(temp(cdindexstart(t):cdindex(t),:,:),1);
end
dindsharedrawrawbetaU = bsxfun(@times,indsharedrawraw,(xv-sum1(cdid,:,:)));
clear temp sum1

% Calculate (partial indsharedrawraw)/(partial betaXZ)...
temp = bsxfun(@times,repmat(xz,[1 draws]),indsharedrawraw);
sum1 = zeros(T,N*draws,numbetaXZ);
for t=1:T
    sum1(t,:,:) = sum(temp(cdindexstart(t):cdindex(t),:,:),1);
end
dindsharedrawrawbetaXZ = bsxfun(@times,indsharedrawraw,(repmat(xz,[1 draws])-sum1(cdid,:,:)));
clear temp sum1

% Calculate (partial indsharedrawraw)/(partial alpha)...
temp = bsxfun(@times,repmat(priceinfo,[1 draws]),indsharedrawraw);
sum1 = zeros(T,N*draws,numalpha);
for t=1:T
    sum1(t,:,:) = sum(temp(cdindexstart(t):cdindex(t),:,:),1);
end
dindsharedrawrawalpha = bsxfun(@times,indsharedrawraw,(repmat(priceinfo,[1 draws])-sum1(cdid,:,:)));
clear temp sum1

% Calculate (partial share)/(partial theta)...
dindsharedrawrawtheta = zeros(cdindex(T),N*draws,numbetaU+numbetaXZ+numalpha);
dindsharedrawrawtheta(:,:,1:numbetaU) = dindsharedrawrawbetaU;
dindsharedrawrawtheta(:,:,numbetaU+1:numbetaU+numbetaXZ) = dindsharedrawrawbetaXZ;
dindsharedrawrawtheta(:,:,numbetaU+numbetaXZ+1:numbetaU+numbetaXZ+numalpha) = dindsharedrawrawalpha;
dsharetheta = permute(sum(bsxfun(@times,repmat(population,[1 draws])./(draws*sum(population,2)),dindsharedrawrawtheta),2),[1 3 2]);

% Calculate (partial delta)/(partial theta)...
ddeltatheta = zeros(cdindex(T),numbetaU+numbetaXZ+numalpha);
n = 0;
for t=1:T
    temp = indsharedrawraw(n+1:cdindex(t),:);
    H1 = bsxfun(@times,repmat(population,[1 draws])./(draws*sum(population,2)),temp)*temp';
    H = diag(sum(bsxfun(@times,repmat(population,[1 draws])./(draws*sum(population,2)),temp),2)) - H1;
    ddeltatheta(n+1:cdindex(t),:) = -H\dsharetheta(n+1:cdindex(t),:);
    n = cdindex(t);
end
clear temp H1 H

% Calculate (d G1)/(d theta)...
dG1theta = (ddeltatheta(1:cdindex(T),:))'*IV./(cdindex(T));


% With (partial delta)/(partial theta) calculated, update
% (partial indsharedrawraw)/(partial theta)

% Calculate true (partial indsharedrawraw)/(partial betaU)...
xvplus = bsxfun(@plus,xv,permute(ddeltatheta(:,1:numbetaU),[1 3 2]));
temp = bsxfun(@times,xvplus,indsharedrawraw);
sum1 = zeros(T,N*draws,numbetaU);
for t=1:T
    sum1(t,:,:) = sum(temp(cdindexstart(t):cdindex(t),:,:),1);
end
truedindsharedrawrawbetaU = bsxfun(@times,indsharedrawraw,(xvplus-sum1(cdid,:,:)));
clear temp sum1

% Calculate true (partial indsharedrawraw)/(partial betaXZ)...
xzplus = bsxfun(@plus,repmat(xz,[1 draws]),permute(ddeltatheta(:,numbetaU+1:numbetaU+numbetaXZ),[1 3 2]));
temp = bsxfun(@times,xzplus,indsharedrawraw);
sum1 = zeros(T,N*draws,numbetaXZ);
for t=1:T
    sum1(t,:,:) = sum(temp(cdindexstart(t):cdindex(t),:,:),1);
end
truedindsharedrawrawbetaXZ = bsxfun(@times,indsharedrawraw,(xzplus-sum1(cdid,:,:)));
clear temp sum1

% Calculate true (partial indsharedrawraw)/(partial alpha)...
priceinfoplus = bsxfun(@plus,repmat(priceinfo,[1 draws]),permute(ddeltatheta(:,numbetaU+numbetaXZ+1:numbetaU+numbetaXZ+numalpha),[1 3 2]));
temp = bsxfun(@times,priceinfoplus,indsharedrawraw);
sum1 = zeros(T,N*draws,numalpha);
for t=1:T
    sum1(t,:,:) = sum(temp(cdindexstart(t):cdindex(t),:,:),1);
end
truedindsharedrawrawalpha = bsxfun(@times,indsharedrawraw,(priceinfoplus-sum1(cdid,:,:)));
clear temp sum1

truedindsharedrawrawtheta = zeros(cdindex(T),N*draws,numbetaU+numbetaXZ+numalpha);
truedindsharedrawrawtheta(:,:,1:numbetaU) = truedindsharedrawrawbetaU;
truedindsharedrawrawtheta(:,:,numbetaU+1:numbetaU+numbetaXZ) = truedindsharedrawrawbetaXZ;
truedindsharedrawrawtheta(:,:,numbetaU+numbetaXZ+1:numbetaU+numbetaXZ+numalpha) = truedindsharedrawrawalpha;



% Calculate (partial G2)/(partial theta)...
dG2theta = zeros(numbetaU+numbetaXZ+numalpha,numbetaXZ);
for i=1:numbetaXZ
    actual = bsxfun(@rdivide,...
        sum(bsxfun(@times,xz(:,:,i),bsxfun(@times,population,indshare)),2),...
        sum(bsxfun(@times,population,indshare)...
        ,2));
    expected = bsxfun(@rdivide,...
        sum(bsxfun(@times,repmat(xz(:,:,i),[1 draws 1]),bsxfun(@times,repmat(population,[1 draws]),truedindsharedrawrawtheta)),2),...
        sum(bsxfun(@times,draws*population,aggshare),2));
    actualtrunc = actual(1:cdindex(T),:,:);
    expectedtrunc = expected(1:cdindex(T),:,:);
    temp = actualtrunc-actualtrunc;
    dG2temptheta =  permute(bsxfun(@minus,temp,expectedtrunc),[1 3 2]);
    dG2theta(:,i) = nanmean(dG2temptheta,1)';
    dG2temptheta(isnan(dG2temptheta)) = 0;
end


% Calculate (partial G3)/(partial theta)...
dG3theta = zeros(numbetaU+numbetaXZ+numalpha,numalpha);
for i=1:numalpha
    actual = bsxfun(@rdivide,...
        sum(bsxfun(@times,priceinfo(:,:,i),bsxfun(@times,population,indshare)),2),...
        sum(bsxfun(@times,population,indshare),2));
    expected = bsxfun(@rdivide,...
        sum(bsxfun(@times,repmat(priceinfo(:,:,i),[1 draws 1]),bsxfun(@times,repmat(population,[1 draws]),truedindsharedrawrawtheta)),2),...
        sum(bsxfun(@times,draws*population,aggshare),2));
    actualtrunc = actual(1:cdindex(T),:,:);
    expectedtrunc = expected(1:cdindex(T),:,:);
    temp = actualtrunc-actualtrunc;
    dG3temptheta =  permute(bsxfun(@minus,temp,expectedtrunc),[1 3 2]);
    dG3theta(:,i) = nanmean(dG3temptheta,1)';
    dG3temptheta(isnan(dG3temptheta)) = 0;
end


% Concatenate moments and return objective function
%
% V is the variance-covariance matrix of the moments. Together with
% the Jacobian matrix of the moments, we can later estimate the 
% variance-covariance matrix of the estimates.


G = [G1;G2;G3];
dGtheta = [dG1theta';dG2theta';dG3theta'];
Vtemp = [G1temp,G2temp,G3temp];
V = (Vtemp'*Vtemp)./cdindex(T);

f = G'/W*G;
grad = 2*dGtheta'/W*G;

% Update estimate, objective function, and gradient value trackers
% Some solvers make repeated calls to the objective function using the same
% theta. If this occurs, the objective function returns the same objective
% function and gradient as before.
lasttheta = theta;
lastf = f;
lastgrad = grad;

end

% end
toc
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