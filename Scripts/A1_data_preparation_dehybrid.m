% A1_data_preparation_dehybrid:
% loads the raw, cleaned data; prepares it for use in the main function;
% and saves it to the hard drive
% Adjusts MPG so that hybrid vehicles are converted into standard vehicles

clear all

% Switch to data folder location and save script folder location
dir_scripts = cd('..\..\Data\2017.10.11\Output');
load('veh_zip_data.mat');
cd(dir_scripts);

% set-up initial parameters

% cdid: specifies which market each vehicle is in
% cdindex: gives the index of the last vehicle in each
% market
cdid = model_yr - 2000;
cdidshift = [1;cdid];
temp = [cdid;9] - cdidshift;
cdindex = find(temp) - 1;
cdindexstart = [0;cdindex(1:7)] + 1;
% clearvars cdidshift temp

N = size(purchasesbyvehzip,2);  % number of zip codes in each market
T = size(cdindex,1);            % number of markets (years)
draws = 3;                      % number of draws for each rand. coef.

num_manu = max(manu_num);       % number of manufacturers in data

% pull the vehicle attributes from the attribute matrix "cars"

mpg = fe08;
price = msrp./10000;
price_low = price;
price_high = price;
price_low(price_low>=1.5) = 0;
price_high(price_high<=9) = 0;
price_low_cons = (price_low>0);
price_outlier_cons = (price_high>0 | price_low>0);
weight = gvw;
numpurchased = numpurchases;
% Luxury makes: Acura, Aston Martin, Audi, Bentley, BMW, Buick, Cadillac,
% Ferrari, Infiniti, Jaguar, Lamborghini, Lexus, Lincoln, Lotus, Maserati,
% Maybach, Mercedes-Benz, Mercury, Porsche, Rolls Royce
luxury = (make_id<=8) | (make_id==13) | (make_id==20) | (make_id==22) | (make_id==25) | (make_id>=27 & make_id<=31) | (make_id>=33 & make_id<=34) | (make_id==43) | (make_id==44); 
% clearvars fe08 msrp gvw numpurchases

mpg_downshift = [0;mpg(1:size(mpg,1)-1)];
% Flags the Honda Insight and the Toyota Prius, the only cars for which
% there only exists a hybrid version
dehybrid_version_dne = (make_id==53 & model_id==504) | (make_id==17 & model_id==220);
dehybrid_mpg_adj_temp = (make_id==17 & model_id==215 & hybrid==1).*(mpg_downshift-mpg)./mpg;
dehybrid_mpg_adj = zeros(T,1);
for t=1:T
    dehybrid_mpg_adj(t) = sum(dehybrid_mpg_adj_temp(cdindexstart(t):cdindex(t),:),1);
end
dehybrid_mpg_adj(dehybrid_mpg_adj==0) = dehybrid_mpg_adj(3);
% clearvars dehybrid_mpg_adj_temp

mpg_dehybrid = mpg;
for t=1:T
    mpg_temp = mpg;
    mpg_temp(hybrid==1,:) = mpg_downshift(hybrid==1,:);
    mpg_temp(hybrid==1 & dehybrid_version_dne==1,:) = mpg(hybrid==1 & dehybrid_version_dne==1,:).*(1+dehybrid_mpg_adj(t));
    mpg_dehybrid(cdindexstart(t):cdindex(t)) = mpg_temp(cdindexstart(t):cdindex(t));
end
% clearvars mpg_temp dehybrid_mpg_adj mpg_downshift

% X: the matrix of non-price vehicle attributes
% C: the matrix of cost shifters, which includes X
X = [ones(cdindex(T),1),10./mpg_dehybrid,liters./10,weight./1000,...
    safety,suv,truck,van,wagon,hybrid,cdid,cdid.^2,luxury,...
    price,price_low,price_high,price_low_cons,price_outlier_cons];
C = [X(:,1:13),imported];

% Set up manufacturer matrices, used for calculating the price elasticity
% matrix, and individual shares, both for subsample and full sample
%
% market: 1 if vehicle and manufacturer are in same market, 0 otherwise
% fullmanu: 1 if vehicle is produced by manufacturer represented by column, 0 otherwise

market = zeros(cdindex(T),num_manu*T);
logic_manu = (repmat([1:num_manu],[cdindex(T) 1]) == repmat(manu_num,[1 num_manu]));
full_manu = zeros(cdindex(T),num_manu*T);
indshare = zeros(cdindex(T),N);

for t=1:T
    market(cdindexstart(t):cdindex(t),num_manu*(t-1)+1:num_manu*t) = ones(cdindex(t)-cdindexstart(t)+1,num_manu);
    full_manu(cdindexstart(t):cdindex(t),num_manu*(t-1)+1:num_manu*t) = logic_manu(cdindexstart(t):cdindex(t),:);
    indshare(cdindexstart(t):cdindex(t),:) = purchasesbyvehzip(cdindexstart(t):cdindex(t),:);
end

% compete: 1 if two vehicles are produced by the same manufacturer, 0
% otherwise
compete = full_manu*full_manu';

% Calculate BLP instruments:
%   non-price cost shifters
%   own_instruments: sum of characteristic over other vehicles 
%       produced by same manufacturer
%   other_instruments: sum of characteristic over competing vehicles
own_instruments = zeros(cdindex(T),4);
other_instruments = zeros(cdindex(T),4);
for i=1:5
    own_instruments(:,i) = sum(bsxfun(@times,sum(bsxfun(@times,X(:,i),full_manu),1),full_manu) - bsxfun(@times,X(:,i),full_manu),2);
    other_instruments(:,i) = sum(bsxfun(@times,sum(bsxfun(@times,X(:,i),(1-full_manu).*market),1),full_manu),2);
end
IV = [C(:,1:14),own_instruments,other_instruments];
% clear own_instruments other_instruments

% pull the zip code demographics from the matrix "cadatazipchar"
population = pop';
popdens = density';
income = reshape(zipinc./10000,[N T])';
% For zip codes where we do not observe mean income, we set mean income to
% the weighted zip-code-average income.
for t=1:T
    incometemp = income(t,:);
    incometemp(incometemp==0) = 6.291913;
    income(t,:) = incometemp;
end
income = income';
income = reshape(income,[1 N*T]);
age65 = age65';
age18 = age18';
gasprice = gaspr';
unemprate = unemplrate';
% clear pop density zipinc gaspr unemplrate incometemp

% Z: matrix of consumer attributes
Z = [age65;age18;unemprate;popdens;gasprice;income];

% Population estimates did not change over our sample period, so we shorten
% the population vector
population = population(1:N);

% Individual and aggregated shares are weighted by the population of each
% zip code
indshare = bsxfun(@rdivide,indshare,population);
aggshare = sum(bsxfun(@times,indshare,population),2)./sum(population);

% Save relevant variables
cd(dir_scripts);
cd('..\Output');
save('prepped_data_dehybrid.mat',...
    'N', 'T', 'draws', 'X', 'C', 'IV', 'price', 'indshare', 'aggshare', ...
    'cdid', 'cdindex', 'cdindexstart', 'Z', 'income','full_manu','compete','population');

% Set current folder to scripts
cd(dir_scripts);

% *EOF* %