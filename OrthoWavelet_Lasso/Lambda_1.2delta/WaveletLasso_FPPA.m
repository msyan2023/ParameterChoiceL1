function [w1,TargetValue] = WaveletLasso_FPPA(c,paraFPPA)
% Solves the following lasso problem via SingleStep FPPA:

%    minimize 1/2*|| Aw - c ||_2^2 + lambda* || w ||_1

%    A is wavelet matrix 

lambda = paraFPPA.lambda;
MaxIter = paraFPPA.MaxIter;
rho = paraFPPA.rho;
beta = paraFPPA.beta;

WaveName = paraFPPA.WaveName;
DecLev = paraFPPA.DecLev;

w1 = zeros(length(c),1);
v1 = zeros(length(c),1);

TargetValue = zeros(1,MaxIter);
% FPPA Iteration
for k = 1:MaxIter
    % w update
    [A_top_v1,L] = wavedec(v1,DecLev,WaveName);    % A'*v1
    w2 = prox_abs(w1-beta*A_top_v1, beta*lambda);
    % v update
    temp = 1/rho*v1 + waverec(2*w2-w1,L,WaveName);  % A*(2*w2-w1)
    v2 = rho*(temp-prox_square(temp,rho,c));
    
    w1 = w2;
    v1 = v2;
    
    TargetValue(k)=1/2*sum((waverec(w1,L,WaveName)-c).^2) + lambda*norm(w1,1);  % Function value
    if rem(k,100)==0
        fprintf('Iter: %d, TargetValue: %f;  \n', k, TargetValue(k))
    end
    
end
disp(' ')
    function y = prox_abs(x,tau)
        %% prox operator for absolute value function $tau*||variable||_1$
        y = sign(x).*max(abs(x)-tau,0);
    end

    function y = prox_square(x,rho,c)
        %% prox operator for square function $1/rho*1/2*||variable-c||_2^2$
        rho1 = 1/rho;
        y = (rho1*c+x)/(rho1+1);
    end

end

