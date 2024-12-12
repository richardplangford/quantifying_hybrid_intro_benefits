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