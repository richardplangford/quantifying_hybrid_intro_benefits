% F1_counterfactual: calculates the compensating variation for all new
% vehicle purchasers who do not have hybrids as a choice and fixes vehicle
% and gas prices

clear all

% global variables which will be sent to the objective function, prevents
%   function from having to load data each time it is called
global lastf lastgrad lasttheta delta tol cdid cdindexstart cdindex N draws T ...
    numbetaXZ numbetaU numalpha betaXZindex aggshare indshare price ...
    X IV Z W C xv xz priceinfo dpriceinfoprice estbetamu estgamma omegahat ...
    compete population incomebrackets outshare V

% load prepared data
dir_scripts = cd('..\Log');
diary('counterfactual.log')
diary on

dir_log = cd('..\..\Data');
load('newcadatacleaned.mat','avggal');

dir_data = cd('..\2022.03.27 Outlier Trim\Output');
load('prepped_data.mat','X');
avggal = avggal(X(:,18)==0,:);
load('full_moment_estimates.mat');

lastn = maxNumCompThreads(2);

% Set up model with estimated coefficients
betaU = full_esttheta(1:numbetaU);
betaO = full_esttheta(numbetaU+1:numbetaU+numbetaXZ);
alpha = full_esttheta(numbetaU+numbetaXZ+1:numbetaU+numbetaXZ+numalpha);

full_manu_orig = full_manu;

% xz: matrix of vehicle and consumer attribute interactions
xz = zeros(cdindex(T),N,numbetaXZ);
for i=1:numbetaXZ
    xzbig = repmat(X(:,betaXZindex(i,1)),[1 N*T]).*repmat(Z(betaXZindex(i,2),:),[cdindex(T) 1]);
    for t=1:T
        xz(cdindexstart(t):cdindex(t),:,i) = xzbig(cdindexstart(t):cdindex(t),N*(t-1)+1:N*t);
    end
    % Replicate 2001 prices across
    if i==5
        for t=1:T
            xz(cdindexstart(t):cdindex(t),:,i) = xzbig(cdindexstart(t):cdindex(t),1:N);
        end
    end
end

% Calculate individual shares from the model
mu = repmat(sum(bsxfun(@times,xz,permute(betaO,[2 3 1])),3),[1 draws]) +...
    sum(bsxfun(@times,xv,permute(betaU,[2 3 1])),3) +...
    repmat(sum(bsxfun(@times,priceinfo,permute(alpha,[2 3 1])),3),[1 draws]);
[indsharedraw,~,aggsharedraw,~] = ...
    mymktsh(delta,mu,N,repmat(population,[1 draws]));
indsharedrawcompressed = sum(reshape(indsharedraw,[cdindex(T) N draws]),3)./draws;
realindshare = indsharedrawcompressed;

% Calculate the Delta matrix, made up of derivatives of shares with 
% respect to price for competing vehicle prices and 0 otherwise
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
    temp = bsxfun(@times,dmudrawprice(:,:,cdindexstart(t):cdindex(t)),bsxfun(@times,-indsharedrawcompressed(cdindexstart(t):cdindex(t),:),permute(indsharedrawcompressed(cdindexstart(t):cdindex(t),:),[3 2 1])));
    temp = temp + bsxfun(@times,permute(eye(cdindex(t)-cdindexstart(t)+1),[1 3 2]),bsxfun(@times,dmudrawprice(:,:,cdindexstart(t):cdindex(t)),permute(indsharedrawcompressed(cdindexstart(t):cdindex(t),:),[3 2 1])));
    Delta(cdindexstart(t):cdindex(t),cdindexstart(t):cdindex(t)) = -permute(bsxfun(@rdivide,sum(bsxfun(@times,population,temp),2),sum(population,2)),[1 3 2]);
end
Delta = Delta.*compete;
mchat = price - Delta\aggsharedraw;
xihat = delta - X*estbetamu;

% mchat = exp(C*estgamma + omegahat)./10000;
mchat_orig = mchat;

% Calculate expected utility under the observed market
eg = exp(bsxfun(@plus,delta,mu));
exp_util = zeros(T,N*draws);
incomebracketsmkt = zeros(T,N,numalpha);
for t=1:T
    exp_util(t,:) = sum(eg(cdindexstart(t):cdindex(t),:),1);
    incomebracketsmkt(t,:,:) = incomebracketstemp(cdindex(t),:,:);
end
exp_util = sum(reshape(exp_util,T,N,draws),3)./draws;

% Remove all hybrids from the market; adjust data accordingly
price_orig = price;
delta_orig = delta;
Xreal = X;
Xtrans = X';
xv = xv(X(:,10)==0,:,:);
xz = xz(X(:,10)==0,:,:);
price = price(X(:,10)==0,:);
IV = IV(X(:,10)==0,:);
delta = delta(X(:,10)==0,:);
compete = compete(X(:,10)==0,Xtrans(10,:)==0);
C = C(X(:,10)==0,:);
mchat = mchat(X(:,10)==0,:);
xihat = xihat(X(:,10)==0,:);
full_manu = full_manu(X(:,10)==0,:);
X = X(X(:,10)==0,:);


realcdindex = cdindex;
realcdindexstart = cdindexstart;
cdid = X(:,11);
cdidshift = [1;cdid];
temp = [cdid;9] - cdidshift;
cdindex = find(temp) - 1;
cdindexstart = [0;cdindex(1:7)] + 1;
clearvars cdidshift temp

% Set up estimated coefficients again
betaU = full_esttheta(1:numbetaU);
betaO = full_esttheta(numbetaU+1:numbetaU+numbetaXZ);
alpha = full_esttheta(numbetaU+numbetaXZ+1:numbetaU+numbetaXZ+numalpha);

i = 0;
norm = 1;
avgnorm = 1;

% % Find new pricing equilibrium by iterating over demand and price solver
% while norm > tol && avgnorm > 1e-3*tol && i < 1000
%     priceinfo = zeros(cdindex(T),N,numalpha);
%     dpriceinfoprice = zeros(cdindex(T),N);
%     incomebrackets = [(income>3.6741 & income<=5.7991);...
%         (income>5.7991 & income<=9.5547);...
%         (income>9.5547)];
%     c = 0;
%     for t=1:T
%         priceinfotemp = repmat(price(cdindexstart(t):cdindex(t),:),[1 N numalpha]);
%         dpriceinfotempprice = repmat(ones(size(price(cdindexstart(t):cdindex(t),:))),[1 N]);
%         priceinfo(cdindexstart(t):cdindex(t),:,:) = priceinfotemp.*repmat(permute(incomebrackets(:,N*(t-1)+1:N*t),[3 2 1]),[cdindex(t)-cdindexstart(t)+1 1 1]);
%         dpriceinfoprice(cdindexstart(t):cdindex(t),:) = dpriceinfotempprice;
%     end
% 
%     % Solve for individual shares at the given prices
%     mudraw = repmat(sum(bsxfun(@times,xz,permute(betaO,[2 3 1])),3),[1 draws]) +...
%         sum(bsxfun(@times,xv,permute(betaU,[2 3 1])),3) +...
%         repmat(sum(bsxfun(@times,priceinfo,permute(alpha,[2 3 1])),3),[1 draws]);
%     [indsharedraw,~,aggsharedraw,~] = ...
%             mymktsh(delta,mudraw,N,repmat(population,[1 draws]));
%     indsharedrawcompressed = sum(reshape(indsharedraw,[cdindex(T) N draws]),3)./draws;
% 
%     % Calculate the Delta matrix, made up of derivatives of shares with 
%     % respect to price for competing vehicle prices and 0 otherwise
%     incomebracketstemp = zeros(cdindex(T),N,numalpha);
%     for t=1:T
%         incomebracketstemp(cdindexstart(t):cdindex(t),:,:) = repmat(permute(incomebrackets(:,N*(t-1)+1:N*t),[3 2 1]),[cdindex(t)-cdindexstart(t)+1 1 1]);
%     end
%     dbetamuprice = estbetamu(14);
%     dmudrawprice = permute(...
%             sum(...
%                 bsxfun(@times,...
%                     bsxfun(@times,permute(alpha,[3 2 1]),incomebracketstemp),...
%                 dpriceinfoprice),...
%             3),...
%         [3 2 1]);
%     dmudrawprice = bsxfun(@plus,dbetamuprice,dmudrawprice);
%     Delta = zeros(cdindex(T),cdindex(T));
%     for t=1:T
%         temp = zeros(cdindex(t)-cdindexstart(t)+1,N,cdindex(t)-cdindexstart(t)+1);
%         temp = bsxfun(@times,dmudrawprice(:,:,cdindexstart(t):cdindex(t)),bsxfun(@times,-indsharedrawcompressed(cdindexstart(t):cdindex(t),:),permute(indsharedrawcompressed(cdindexstart(t):cdindex(t),:),[3 2 1])));
%         temp = temp + bsxfun(@times,permute(eye(cdindex(t)-cdindexstart(t)+1),[1 3 2]),bsxfun(@times,dmudrawprice(:,:,cdindexstart(t):cdindex(t)),permute(indsharedrawcompressed(cdindexstart(t):cdindex(t),:),[3 2 1])));
%         Delta(cdindexstart(t):cdindex(t),cdindexstart(t):cdindex(t)) = -permute(bsxfun(@rdivide,sum(bsxfun(@times,population,temp),2),sum(population,2)),[1 3 2]);
%     end
%     Delta = Delta.*compete;
%     
%     % Calculate prices using given market shares
%     new_price = mchat + Delta\aggsharedraw;
%     new_delta = [X(:,1:13),new_price,X(:,15:18)]*estbetamu + xihat;
% 
%     % Calculate convergence criterion
%     clear temp
%     w = abs(new_price - price);
%     norm = max(w);
%     avgnorm = mean(w);
%     price = new_price;
%     delta = new_delta;
%     i = i+1;
% end

% Set up model to calculate 
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

% Calculate counterfactual individual shares and expected utility
mu_cf = repmat(sum(bsxfun(@times,xz,permute(betaO,[2 3 1])),3),[1 draws]) +...
    sum(bsxfun(@times,xv,permute(betaU,[2 3 1])),3) +...
    repmat(sum(bsxfun(@times,priceinfo,permute(alpha,[2 3 1])),3),[1 draws]);
[indsharedraw,~,aggsharedraw,~] = ...
    mymktsh(delta,mu_cf,N,repmat(population,[1 draws]));
indsharedrawcompressed = sum(reshape(indsharedraw,[cdindex(T) N draws]),3)./draws;
eg_cf = exp(bsxfun(@plus,delta,mu_cf));
exp_util_cf = zeros(T,N*draws);
incomebracketsmkt = zeros(T,N,numalpha);
for t=1:T
    exp_util_cf(t,:) = sum(eg_cf(cdindexstart(t):cdindex(t),:),1);
    incomebracketsmkt(t,:,:) = incomebracketstemp(cdindex(t),:,:);
end
exp_util_cf = sum(reshape(exp_util_cf,T,N,draws),3)./draws;

realoutshare = zeros(T,N);
outshare = zeros(T,N);
for t=1:T
    realoutshare(t,:) = 1-sum(realindshare(realcdindexstart(t):realcdindex(t),:),1);
    outshare(t,:) = 1-sum(indsharedrawcompressed(cdindexstart(t):cdindex(t),:),1);
end

% Use observed and counterfactual utility to calculate compensating
% variation
% NOTE: Compensating variation is calculated without outside option.
% Numbers apply only to new vehicle purchasers.
% 2010 Avg. CPI: 218.056
% 2017 Avg. CPI: 245.120
% Measured 01/28/2020
% Used to convert 2010$ to 2017$
incomebracketsmkttemp = zeros(T,N,numalpha+1);
incomebracketsmkttemp(:,:,2:4) = incomebracketsmkt;
incomebracketsmkttemp(:,:,1) = 1-sum(incomebracketsmkt,3);
incomebracketsmkt = incomebracketsmkttemp;

comp_var = -bsxfun(@rdivide,log(exp_util./exp_util_cf),permute(bsxfun(@plus,estbetamu(14),[0;alpha])./10000,[2 3 1])).*245.120./218.056;
average_cv_by_incomebracket = permute(sum(bsxfun(@times,incomebracketsmkt.*comp_var,bsxfun(@times,(1-outshare),population)),2)./sum(bsxfun(@times,incomebracketsmkt,bsxfun(@times,(1-outshare),population)),2),[1 3 2]);
average_cv = bsxfun(@rdivide,sum(bsxfun(@times,sum(incomebracketsmkt.*comp_var,3),bsxfun(@times,(1-outshare),population)),2),sum(bsxfun(@times,(1-outshare),population),2));

% Calculate weighted-average VMT and GPM
% First, calculate VMT, GPM, and gallons (using average VMT for each
% vehicle) times individual vehicle share
realmilesweightedtemp = bsxfun(@times,realindshare,avggal(:,1));
realsharetemp = bsxfun(@rdivide,realmilesweightedtemp,avggal(:,1));
realgpmweightedtemp = bsxfun(@times,realindshare,Xtrans(2,:)');
realgalweightedtemp = bsxfun(@times,realindshare,avggal(:,1).*Xtrans(2,:)'./10);
realmiles = zeros(T,N);
realgpm = zeros(T,N);
realgal = zeros(T,N);
for t=1:T
    % Pull out the relevant market
    realmilesweightedchunk = realmilesweightedtemp(realcdindexstart(t):realcdindex(t),:);
    realgpmweightedchunk = realgpmweightedtemp(realcdindexstart(t):realcdindex(t),:);
    realsharechunk = realsharetemp(realcdindexstart(t):realcdindex(t),:);
    realgalweightedchunk = realgalweightedtemp(realcdindexstart(t):realcdindex(t),:);
    
    % Calculate the share-weighted miles traveled and gallons used
    % for each individual in each market
    % First, calculate observed weighted average
    realmiles(t,:) = nansum(realmilesweightedchunk,1)./nansum(realsharechunk,1);
    % Then, insert the observed market average times individual vehicle share
    % where unobserved
    realmilesweightedchunk(isnan(realmilesweightedchunk)) = 0;
    realmilesweightedchunk = realmilesweightedchunk + isnan(realmilesweightedtemp(realcdindexstart(t):realcdindex(t),:)).*bsxfun(@times,realindshare(realcdindexstart(t):realcdindex(t),:),realmiles(t,:));
    realgal(t,:) = nansum(realgalweightedchunk,1)./nansum(realsharechunk,1);
    realgalweightedchunk(isnan(realgalweightedchunk)) = 0;
    realgalweightedchunk = realgalweightedchunk + isnan(realgalweightedtemp(realcdindexstart(t):realcdindex(t),:)).*bsxfun(@times,realindshare(realcdindexstart(t):realcdindex(t),:),realgal(t,:));
    
    % Then, calculate the share-weighted market average for miles traveled,
    % GPM, and gallons used for each individual
    realmiles(t,:) = sum(realmilesweightedchunk,1)./sum(realindshare(realcdindexstart(t):realcdindex(t),:),1);
    realgpm(t,:) = sum(realgpmweightedchunk,1)./sum(realindshare(realcdindexstart(t):realcdindex(t),:),1);
    realgal(t,:) = sum(realgalweightedchunk,1)./sum(realindshare(realcdindexstart(t):realcdindex(t),:),1);
end
% Calculate population-weighted market average of gallons used for each
% market
realgalavg = bsxfun(@rdivide,sum(population.*realgal,2),sum(population,2));
% Alternatively, calculate by using individual market-average miles
% traveled times individual market average GPM, then calculating
% population-weighted average
alt_realgal = realmiles.*realgpm./10;
alt_realgalavg = bsxfun(@rdivide,sum(population.*alt_realgal,2),sum(population,2));

% Now, repeat all of the above for the counterfactual where hybrids are
% removed
avggal_cf = avggal(X(:,10)==0,:);
% Calculate weighted-average VMT and GPM for the counterfactual
cfmilesweightedtemp = bsxfun(@times,indsharedrawcompressed,avggal_cf(:,1));
cfsharetemp = bsxfun(@rdivide,cfmilesweightedtemp,avggal_cf(:,1));
cfgpmweightedtemp = bsxfun(@times,indsharedrawcompressed,X(:,2));
cfgalweightedtemp = bsxfun(@times,indsharedrawcompressed,avggal_cf(:,1).*X(:,2)./10);
cfmiles = zeros(T,N);
cfgpm = zeros(T,N);
cfgal = zeros(T,N);
for t=1:T
    cfmilesweightedchunk = cfmilesweightedtemp(cdindexstart(t):cdindex(t),:);
    cfgpmweightedchunk = cfgpmweightedtemp(cdindexstart(t):cdindex(t),:);
    cfsharechunk = cfsharetemp(cdindexstart(t):cdindex(t),:);
    cfgalweightedchunk = cfgalweightedtemp(cdindexstart(t):cdindex(t),:);
    
    cfmiles(t,:) = nansum(cfmilesweightedchunk,1)./nansum(cfsharechunk,1);
    cfmilesweightedchunk(isnan(cfmilesweightedchunk)) = 0;
    cfmilesweightedchunk = cfmilesweightedchunk + isnan(cfmilesweightedtemp(cdindexstart(t):cdindex(t),:)).*bsxfun(@times,indsharedrawcompressed(cdindexstart(t):cdindex(t),:),cfmiles(t,:));
    cfgal(t,:) = nansum(cfgalweightedchunk,1)./nansum(cfsharechunk,1);
    cfgalweightedchunk(isnan(cfgalweightedchunk)) = 0;
    cfgalweightedchunk = cfgalweightedchunk + isnan(cfgalweightedtemp(cdindexstart(t):cdindex(t),:)).*bsxfun(@times,indsharedrawcompressed(cdindexstart(t):cdindex(t),:),cfgal(t,:));
    
    cfmiles(t,:) = sum(cfmilesweightedchunk,1)./sum(indsharedrawcompressed(cdindexstart(t):cdindex(t),:),1);
    cfgpm(t,:) = sum(cfgpmweightedchunk,1)./sum(indsharedrawcompressed(cdindexstart(t):cdindex(t),:),1);
    cfgal(t,:) = sum(cfgalweightedchunk,1)./sum(indsharedrawcompressed(cdindexstart(t):cdindex(t),:),1);
end
cfgalavg = bsxfun(@rdivide,sum(population.*cfgal,2),sum(population,2));
% In addition, calculate a counterfactual based on real share-weighted
% market average miles for each individual instead of the counterfactual
% (in case miles traveled is individual-specific rather than
% vehicle-specific)
alt_cfgal = cfmiles.*cfgpm./10;
alt_cfgalavg = bsxfun(@rdivide,sum(population.*alt_cfgal,2),sum(population,2));
altfixed_cfgal = realmiles.*cfgpm./10;
altfixed_cfgalavg = bsxfun(@rdivide,sum(population.*altfixed_cfgal,2),sum(population,2));

% Set up counterfactual price, MC, and individual shares inside
% original-size variables
indsharedrawcompressedtemp = zeros(size(realindshare));
pricetemp = zeros(size(price_orig));
mchattemp = zeros(size(mchat));
indsharedrawcompressedtemp(Xreal(:,10)==0,:) = indsharedrawcompressed;
pricetemp(Xreal(:,10)==0,:) = price;
mchattemp(Xreal(:,10)==0,:) = mchat;
indsharedrawcompressed = indsharedrawcompressedtemp;
price = pricetemp;
mchat = mchattemp;
clear indsharedrawcompressedtemp pricetemp mchattemp

% Calculating real and counterfactual outside good shares
realoutshare = zeros(T,N);
outshare = zeros(T,N);
for t=1:T
    realoutshare(t,:) = 1-sum(realindshare(realcdindexstart(t):realcdindex(t),:),1);
    outshare(t,:) = 1-sum(indsharedrawcompressed(realcdindexstart(t):realcdindex(t),:),1);
end

% Calculating social cost of carbon
% 18.9 lbs CO2/gallon of gasoline
% 2000 lbs/ton
% $45/ton of CO2 = social cost of carbon in 2017$
scc = (altfixed_cfgalavg - alt_realgalavg).*18.9./2000.*45;
% Version where miles traveled varies in counterfactual
scc_vary = (alt_cfgalavg - alt_realgalavg).*18.9./2000.*45;
% Summing across individuals to get total SCC
total_scc = sum(bsxfun(@times,1-outshare,population).*(altfixed_cfgal - alt_realgal).*18.9./2000.*45,2);
% Summing across individuals to get total compensating variation
total_comp_var = sum(bsxfun(@times,sum(incomebracketsmkt.*comp_var,3),bsxfun(@times,1-outshare,population)),2);

type_mat = zeros(realcdindex(T),7*T);
for t=1:T
    type_mat(realcdindexstart(t):realcdindex(t),7*(t-1)+1:7*t) = [Xreal(realcdindexstart(t):realcdindex(t),6:10),(1-sum(Xreal(realcdindexstart(t):realcdindex(t),6:9),2)<=0),ones(realcdindex(t)-realcdindexstart(t)+1,1)];
end
type_mat_cf = zeros(cdindex(T),7*T);
for t=1:T
    type_mat_cf(cdindexstart(t):cdindex(t),7*(t-1)+1:7*t) = [X(cdindexstart(t):cdindex(t),6:10),(1-sum(X(cdindexstart(t):cdindex(t),6:9),2)<=0),ones(cdindex(t)-cdindexstart(t)+1,1)];
end

num_manu = size(full_manu,2)./T;
% Calculate the population-weighted share change between real and
% counterfactual
share_change = bsxfun(@rdivide,sum(population.*(indsharedrawcompressed - realindshare),2),sum(population,2));
% Calculate the share change by manufacturer
share_change_agg = reshape((share_change'*full_manu_orig)',[num_manu T])';
share_change_type_agg = reshape((share_change'*type_mat)',[7 T])';
% Calculate share change by manufacturer conditional on (population-weighted average) vehicle purchase
relative_share_change_agg = bsxfun(@rdivide,share_change_agg,bsxfun(@rdivide,(sum(bsxfun(@times,1-outshare,population),2)),(sum(population,2))));
relative_share_change_type_agg_mat = mean(bsxfun(@rdivide,share_change_type_agg,bsxfun(@rdivide,(sum(bsxfun(@times,1-outshare,population),2)),(sum(population,2)))),1)';

share_change_cf = share_change(Xreal(:,10)==0);
share_change_agg_cf = reshape((share_change_cf'*full_manu)',[num_manu T])';
share_change_type_agg_cf = reshape((share_change_cf'*type_mat_cf)',[7 T])';
relative_share_change_agg_cf = bsxfun(@rdivide,share_change_agg_cf,bsxfun(@rdivide,(sum(bsxfun(@times,1-outshare,population),2)),(sum(population,2))));
relative_share_change_type_agg_cf_mat = mean(bsxfun(@rdivide,share_change_type_agg_cf,bsxfun(@rdivide,(sum(bsxfun(@times,1-outshare,population),2)),(sum(population,2)))),1)';

% Calculate change in profits for each manufacturer (in 2017$)
realprofits = zeros(T,num_manu);
profits_cf = zeros(T,num_manu);
for t=1:T
    manu = full_manu_orig(realcdindexstart(t):realcdindex(t),(t-1)*num_manu+1:t*num_manu);
    realrev = sum(bsxfun(@times,realindshare(realcdindexstart(t):realcdindex(t),:),population).*sum(price_orig(realcdindexstart(t):realcdindex(t),:,:),3),2)*10000.*244.786./218.056;
    realcost = sum(bsxfun(@times,realindshare(realcdindexstart(t):realcdindex(t),:),population).*repmat(mchat_orig(realcdindexstart(t):realcdindex(t),:),[1 N]),2)*10000.*244.786./218.056;
    realprofits(t,:) = (realrev - realcost)'*manu;
    rev_cf = sum(bsxfun(@times,indsharedrawcompressed(realcdindexstart(t):realcdindex(t),:),population).*sum(price(realcdindexstart(t):realcdindex(t),:,:),3),2)*10000.*244.786./218.056;
    cost_cf = sum(bsxfun(@times,indsharedrawcompressed(realcdindexstart(t):realcdindex(t),:),population).*repmat(mchat(realcdindexstart(t):realcdindex(t),:),[1 N]),2)*10000.*244.786./218.056;
    profits_cf(t,:) = (rev_cf - cost_cf)'*manu;
end
total_ps = sum(realprofits - profits_cf,2);

% Add a rebound effect based on a -0.15 elasticity of driving w.r.t. price
% per mile of driving (or GPM, since the PPG cancels out in equation below)
altfixed_cfmiles_adj = (((cfgpm - realgpm)./realgpm).*-0.15).*realmiles + realmiles;
altfixed_cfgal_adj_2 = altfixed_cfmiles_adj.*cfgpm./10;
altfixed_cfgalavg_adj_2 = bsxfun(@rdivide,sum(population.*altfixed_cfgal_adj_2,2),sum(population,2));
scc_adj = (altfixed_cfgalavg_adj_2 - alt_realgalavg).*18.9./2000.*45;
total_scc_adj = sum(bsxfun(@times,1-outshare,population).*(altfixed_cfgal_adj_2 - alt_realgal).*18.9./2000.*45,2);


% % Calculate change in shares
% shares_change = bsxfun(@rdivide,sum(population.*indsharedrawcompressed,2),sum(population,2))-bsxfun(@rdivide,sum(population.*realindshare(X(:,10)==0,:),2),sum(population,2));
% change_results = [(1:size(shares_change,1))',shares_change,cdid];
% xlswrite('\\bateswhite.com\dc-fs\Home\rlangford\My Documents\MATLAB\Langford Gillingham 2015 MATLAB Work\2016.10.25\Output\shares_change.xlsx',change_results);


save('counterfactuals_fixed_veh_gas_prices.mat');

% save('counterfactual_results.mat',...
%     'N', 'T', 'draws', 'X', 'C', 'IV', 'price', 'indshare', 'aggshare', ...
%     'cdid', 'cdindex', 'Z', 'income','full_manu','compete','population', ...
%     'betaUindex', 'betaXZindex', 'estbetamu', 'full_esttheta', ...
%     'comp_var', 'average_cv_by_incomebracket', 'average_cv', 'exp_util', ...
%     'exp_util_cf', 'incomebracketsmkt');
% cd(dir_scripts);








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