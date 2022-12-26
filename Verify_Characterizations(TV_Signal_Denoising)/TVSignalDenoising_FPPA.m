function [w1,B_u,TargetValue] = TVSignalDenoising_FPPA(A,c,paraFPPA)
% Solves the following ROF model via SingleStep FPPA:

% minimize 1/2*|| w - c ||_2^2 + lambda* || Aw ||_1

% A is first order difference matrix (applying on one dimensional signal)

lambda = paraFPPA.lambda;
MaxIter = paraFPPA.MaxIter;
rho = paraFPPA.rho;
beta = paraFPPA.beta;

[nRowA, nColA]=size(A);

w1 = zeros(nColA,1);
v1 = zeros(nRowA,1);

TargetValue = zeros(1,MaxIter);
% FPPA Iteration
for k = 1:MaxIter
    % w update
    w2 = prox_square(w1-beta*FirstOrderDiff_T(v1), beta,c);
 
    % v update
    temp = 1/rho*v1 + FirstOrderDiff(2*w2-w1);
    v2 = rho*(temp-prox_abs(temp,lambda/rho));
    
    w1 = w2;
    v1 = v2;
    
    TargetValue(k)=1/2*sum((w1 - c).^2) + lambda*norm(FirstOrderDiff(w1),1);  %  Function value
    if rem(k,1000)==0
        fprintf('Iter: %d, TargetValue: %f;  \n', k, TargetValue(k))
    end
end
B_u = prox_abs(temp,lambda/rho);
    function y = prox_abs(x,tau)
        %% prox operator for absolute value function $tau*||variable||_1$
        y = sign(x).*max(abs(x)-tau,0);
    end

    function y = prox_square(t,beta,c)
        %% prox operator for square function $beta*1/2*||variable-c||_2^2$
        y = (beta*c+t)/(beta+1);
    end

    function y = FirstOrderDiff(x)
        y = x(2:end) - x(1:end-1);
    end

    function y = FirstOrderDiff_T(x)
        y = [-x(1);
              x(1:end-1) - x(2:end);
              x(end)];
    end

end

