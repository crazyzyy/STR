%% Main code of STR method 
% technique details see https://www.frontiersin.org/articles/10.3389/fncom.2017.00101/full

%% Discription of input
% Vi (vector: 1*L):  voltage time series of target neuron.
%                    Recommandation: preprocessing the original recording of voltage to a series of time interval length dt~0.5ms
%                    Note: Vi[k], Si[k], S[k] are signals at time step at (k-1)*dt
% Ji (vector: 1*L):  binary array, Ji=1 for the subthreshold region of Vi.
% S  (matrix: N*L):  binary spike train time series (other than Si) obtained from MEA 
% p1 (scalar)     :  regression order for Vi
% p2 (scalar)     :  regression order for ST
%                    Ji[k] = 1 when Vi[k] is at the subthreshold region, otherwise 0. 
% r  (scalar)     :  pvalue for significance test (see paper)

%% Discription of Output
% fields of Rslt: 
% [hyper parameter][input] p1,p2,r, ratio (threshold for significance test, computed from r and p2)
% [regression parameter] b0 (\beta^0 in paper), b (\beta^k in paper), a (alpha_j^l in paper)
% [prediction error] res_as (res for Ji=1, 0 for Ji=0)
% [standard deviation of reg param] vb (std of b), va (std of a)
% [statistical quantities for inference] a_tl, M, M_tl, lij (see paper)
% [reconstructed network connectivity] ExCpl, InCpl,  Aj2i(adjacency array. 1 for Ex, -1 for In)
% ExCpl (or InCpl) = [indices of coupled presynaptic neurons, indicator of coupling strength, chance of false positive]
% [information criteria] ic (ic.AIC, ic.BIC can be used for the selection of p1 and p2)

%% Comments about the input
% 1. Vi,Ji,S recommanded sampling interval length (after preprocessing) ~0.5ms 
% (all the following comments are based on 0.5ms interval!)
% 2. Ji, if the target neuron fires sparsely, it is recommanded to remove more
% (say -5ms before spike to 5~10ms after spike) from the subthreshold region
% 3. p1: 5~10. Recommand to change p1 from 1 to 20 and plot the BIC values.
% 4. p2: 5~30. When p1 is determined, change p2 from 1 to 30 and plot the BIC values.
% 5. r : ~0.01, depending on the desired false positive rate. Weak
% couplings may be ignored if r is too small.

%% Code
function [Rslt] = STR_main(Vi,Ji,S,p1,p2,r)
    Rslt = [];
    % hyper parameters
    % regression order
    Rslt.reg_od = [p1,p2];
    % for the determination of subthreshold region Ji
    % pvalue
    Rslt.r = r;
    
    N = size(S,1);
    % exclude the first max(p1,p2) points of Vi for regression
    Ji(1:max(p1,p2)) = 0;
    Ji = (Ji==1);
    % indices of Vi belonging to Ji
    idx = find(Ji); 
    L_eff  = length(idx);
    
    % 1*L_eff
    V_rg   = Vi(idx);
    % p1*L_eff
    V_hstr = cell2mat(arrayfun(@(x) Vi(idx-x),(1:p1)','UniformOutput', false));
    % (p2N)*L_eff
    S_hstr = cell2mat(arrayfun(@(x) S(:,idx-x),(1:p2)','UniformOutput', false)); 
    % (p1+p2N+1)*L_eff
    rg_terms = [V_hstr;S_hstr;ones(1,L_eff)];
    
    % Linear Regression Step
    [pm,sE,ic,res] = LR(rg_terms',V_rg');

    % b0: beta_i^0 in paper (i fixed)
    Rslt.b0 = pm(end);
    % b : beta_i^k in paper (i fixed)
    Rslt.b  = pm(1:p1);
    Rslt.vb = sE(1:p1);

    % a (matrix: N*p2): alpha_ij^l in paper (i fixed)
    a  = reshape(pm(p1+1:end-1),N,p2);
    va = reshape(sE(p1+1:end-1),N,p2);   
    % a_tl (vector: N*1): alpha_tilde in paper
    a_tl = a./va;
    
    [~,lij] = max(abs(a_tl),[],2);
    M    = arrayfun(@(x) a(x,lij(x)),(1:N)');
    M_tl = arrayfun(@(x) a_tl(x,lij(x)),(1:N)');
        
    Rslt.M = M;
    Rslt.M_tl = M_tl;
    Rslt.lij = lij;
    Rslt.a = a;
    Rslt.va = va;
    Rslt.a_tl = a_tl;
    Rslt.ic = ic;
    
    ratio = norminv(1-r/2/p2);
    Aj2i = sign(M).*(abs(M_tl)>ratio);
    % [indices of coupled presynaptic neurons, indicator of coupling strength, chance of false positive]
    Rslt.ExCpl = [find(Aj2i>0), M(Aj2i>0),normcdf(-abs(M_tl(Aj2i>0)))*2*p2];
    Rslt.InCpl = [find(Aj2i<0), M(Aj2i<0),normcdf(-abs(M_tl(Aj2i<0)))*2*p2];
    % adjacency array
    Rslt.Aj2i  = Aj2i;
    Rslt.ratio = ratio;
    
    res_as = zeros(size(Vi));
    res_as(Ji) = res;
    Rslt.res_as = res_as;

end


function [Ji] = S2J(Si,rm_len)
    rm_len_l = rm_len(1); %left
    rm_len_r = rm_len(2); %right
    rm_total = rm_len_l+rm_len_r+1;
    Ji = conv(Si,ones(1,rm_total));
    Ji = ~Ji(rm_len_l+1:end-rm_len_r);
end

%YY (matrix: n*k)
%Y  (vector: n*1)
%linear regression
function [b,sE,ic,res] = LR(YY,Y)
    [n,k] = size(YY);
    A = YY'*YY;
    b = A\(YY'*Y);
    
    Ypd = YY*b;
    res = Y - Ypd;
    V = var(res);
    
    %ic: information criteria
    ic = [];
    ic.n = n;ic.k = k;ic.v = V;
    % see https://en.wikipedia.org/wiki/Bayesian_information_criterion 
    % Gaussian special case
    ic.BIC = (n*log(V)+k*log(n))/n;
    % see https://en.wikipedia.org/wiki/Akaike_information_criterion
    ic.AIC = (n*log(V)+2*k)/n;

    U2 =  YY.*repmat(res,1,k);
    U22 = U2'*U2;
    Hs = A\U22/A';
    % sE: sigma_j in paper
    sE = sqrt(diag(Hs));
end
