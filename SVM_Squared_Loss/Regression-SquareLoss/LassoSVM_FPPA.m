function [w1,TargetValue] = LassoSVM_FPPA(A,c,paraFPPA)
% Solves the following lasso problem via SingleStep FPPA:

%    minimize 1/2*|| Aw - c ||_2^2 + lambda || w(1:end-1) ||_1
%                   w = [alpha b]


lambda = paraFPPA.lambda;
MaxIter = paraFPPA.MaxIter;
rho = paraFPPA.rho;
beta = paraFPPA.beta;

n=size(A,2);

w1 = zeros(n,1);
v1 = zeros(n-1,1);

TargetValue = zeros(1,MaxIter);
% FPPA Iteration
for k = 1:MaxIter
    % w update
    w2 = prox_abs(w1-beta*A.'*v1, beta*lambda);
    
    % v update
    temp = 1/rho*v1 + A*(2*w2-w1);
    v2 = rho*(temp-prox_square(temp,rho,c));
    
    w1 = w2;
    v1 = v2;
    
    TargetValue(k)=1/2*sum((A*w1 - c).^2) + lambda*norm(w1(1:end-1),1);  % Function value
    if rem(k,1000)==0
        fprintf('Iter: %d, TargetValue: %f;  ', k, TargetValue(k))
    end
end

    function y = prox_abs(x,tau)
        %% prox operator for absolute value function $tau*||variable(1:end-1)||_1$
        y = sign(x).*max(abs(x)-tau,0);
        y(end) = x(end);
    end

    function y = prox_square(x,rho,c)
        %% prox operator for square function $1/rho*1/2*||variable-c||_2^2$
        rho1 = 1/rho;
        y = (rho1*c+x)/(rho1+1);
    end

end

