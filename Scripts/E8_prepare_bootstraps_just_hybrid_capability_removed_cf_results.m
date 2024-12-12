% B0_model_estimation: sets up and runs the objective function solver to obtain estimates
clear all

cd('..\Input');
load('bootstrap_mat.mat');

num_bootstraps = size(bootstrap_mat_group,2);

estbetamu_mat = zeros(12,num_bootstraps);
esttheta_mat = zeros(12,num_bootstraps);
estgamma_mat = zeros(14,num_bootstraps);

avg_cv_1 = zeros(8,num_bootstraps);
avg_cv_2 = zeros(8,num_bootstraps);
avg_cv_3 = zeros(8,num_bootstraps);
avg_cv_4 = zeros(8,num_bootstraps);
average_cv_mat = zeros(8,num_bootstraps);
total_cv_mat = zeros(8,num_bootstraps);
scc_mat = zeros(8,num_bootstraps);
vary_scc_mat = zeros(8,num_bootstraps);
total_scc_mat = zeros(8,num_bootstraps);
total_ps_mat = zeros(8,num_bootstraps);

scc_adj_mat = zeros(8,num_bootstraps);
total_scc_adj_mat = zeros(8,num_bootstraps);

relative_share_change_type_agg_cf_mat = zeros(7,num_bootstraps);
relative_share_change_type_agg_mat = zeros(7,num_bootstraps);

for cf_run2=1:num_bootstraps

cd('..\Input');
load('bootstrap_mat.mat');
    
% load prepared data  
group = bootstrap_mat_group(cf_run2);
run = bootstrap_mat_run(cf_run2);

disp(['Group ',num2str(group),', Run ',num2str(run),':']);

cd('..\Output');
fileloc = sprintf('counterfactual_just_hybrid_capability_removed_group%d_run%d.mat',group,run);
load(fileloc);

estbetamu_mat(:,cf_run2) = estbetamu;
esttheta_mat(:,cf_run2) = full_esttheta;
estgamma_mat(:,cf_run2) = estgamma;

avg_cv_1(:,cf_run2) = average_cv_by_incomebracket(:,1);
avg_cv_2(:,cf_run2) = average_cv_by_incomebracket(:,2);
avg_cv_3(:,cf_run2) = average_cv_by_incomebracket(:,3);
avg_cv_4(:,cf_run2) = average_cv_by_incomebracket(:,4);
average_cv_mat(:,cf_run2) = bsxfun(@rdivide,sum(bsxfun(@times,bsxfun(@times,sum(incomebracketsmkt.*comp_var,3),(1-outshare)),population),2),sum(bsxfun(@times,(1-outshare),population),2));
total_cv_mat(:,cf_run2) = sum(bsxfun(@times,bsxfun(@times,sum(incomebracketsmkt.*comp_var,3),(1-outshare)),population),2);
scc_mat(:,cf_run2) = scc;
vary_scc_mat(:,cf_run2) = (alt_cfgalavg - alt_realgalavg).*18.9./2000.*45;
total_scc_mat(:,cf_run2) = total_scc;
total_ps_mat(:,cf_run2) = sum(realprofits - profits_cf,2);


altfixed_cfmiles_adj = (((cfgpm - realgpm)./realgpm).*-0.15).*realmiles + realmiles;
altfixed_cfgal_adj_2 = altfixed_cfmiles_adj.*cfgpm./10;
altfixed_cfgalavg_adj_2 = bsxfun(@rdivide,sum(population.*altfixed_cfgal_adj_2,2),sum(population,2));

scc_adj_mat(:,cf_run2) = (altfixed_cfgalavg_adj_2 - alt_realgalavg).*18.9./2000.*45;
total_scc_adj_mat(:,cf_run2) = sum(bsxfun(@times,1-outshare,population).*(altfixed_cfgal_adj_2 - alt_realgal).*18.9./2000.*45,2);






type_mat = zeros(realcdindex(T),7*T);
for t=1:T
    type_mat(realcdindexstart(t):realcdindex(t),7*(t-1)+1:7*t) = [Xreal(realcdindexstart(t):realcdindex(t),6:10),(1-sum(Xreal(realcdindexstart(t):realcdindex(t),6:9),2)<=0),ones(realcdindex(t)-realcdindexstart(t)+1,1)];
end
type_mat_cf = zeros(cdindex(T),7*T);
for t=1:T
    type_mat_cf(cdindexstart(t):cdindex(t),7*(t-1)+1:7*t) = [X(cdindexstart(t):cdindex(t),6:10),(1-sum(X(cdindexstart(t):cdindex(t),6:9),2)<=0),ones(cdindex(t)-cdindexstart(t)+1,1)];
end

num_manu = size(full_manu,2)./T;
share_change = bsxfun(@rdivide,sum(population.*(indsharedrawcompressed - realindshare),2),sum(population,2));
share_change_agg = reshape((share_change'*full_manu_orig)',[num_manu T])';
share_change_type_agg = reshape((share_change'*type_mat)',[7 T])';
relative_share_change_agg = bsxfun(@rdivide,share_change_agg,bsxfun(@rdivide,(sum(bsxfun(@times,1-outshare,population),2)),(sum(population,2))));
relative_share_change_type_agg_mat(:,cf_run2) = mean(bsxfun(@rdivide,share_change_type_agg,bsxfun(@rdivide,(sum(bsxfun(@times,1-outshare,population),2)),(sum(population,2)))),1)';

share_change_cf = share_change;
share_change_agg_cf = reshape((share_change_cf'*full_manu)',[num_manu T])';
share_change_type_agg_cf = reshape((share_change_cf'*type_mat_cf)',[7 T])';
relative_share_change_agg_cf = bsxfun(@rdivide,share_change_agg_cf,bsxfun(@rdivide,(sum(bsxfun(@times,1-outshare,population),2)),(sum(population,2))));
relative_share_change_type_agg_cf_mat(:,cf_run2) = mean(bsxfun(@rdivide,share_change_type_agg_cf,bsxfun(@rdivide,(sum(bsxfun(@times,1-outshare,population),2)),(sum(population,2)))),1)';


end