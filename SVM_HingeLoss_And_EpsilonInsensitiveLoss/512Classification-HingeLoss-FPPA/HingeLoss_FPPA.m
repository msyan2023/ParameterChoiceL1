function [w1, v1, TargetValue] = HingeLoss_FPPA(A, paraFPPA)
%---------------------------------------------------------------------------
%    min phi(w) + psi(Aw), w = [alpha b]
%    For classification with hinge loss, 
%    phi is regularization term,  phi(w) = lambda*|w(1:end-1)|.
%    psi is epsilon-insensitive hinge loss,  psi_b(w) = max(1-w,0).       
%---------------------------------------------------------------------------                
lambda = paraFPPA.lambda;
MaxIter = paraFPPA.MaxIter;
rho = paraFPPA.rho;
beta = paraFPPA.beta;

n=size(A,2); 
if isfield(paraFPPA,'initial_w') && isfield(paraFPPA, 'initial_v')
    w1 = paraFPPA.initial_w;
    v1 = paraFPPA.initial_v;
else
    w1 = zeros(n,1);
    v1 = zeros(n-1,1);
end

TargetValue = zeros(1,MaxIter);
% FPPA Iteration
for k = 1:1:MaxIter
    % w update
    w2 = prox_abs(w1-beta*A.'*v1, beta*lambda);
    % v update
    temp = 1/rho*v1 + A*(2*w2-w1);
    v2 = rho*(temp-prox_hinge(temp,rho));
    
    w1 = w2;
    v1 = v2;
    
    % record
    TargetValue(k) = lambda*norm(w1(1:end-1),1) + sum(hinge(A*w1));  % Function value
    if rem(k,1000)==0
        fprintf('Iter: %d, TargetValue: %f;  \n', k, TargetValue(k))
    end
end


    function y = prox_abs(x,tau)
        %% prox operator for absolute value function $tau*||x(1:end-1)||_1$
        y = sign(x).*max(abs(x)-tau,0);
        y(end) = x(end);
    end

    function y = prox_hinge(x,rho)
        %% prox operator for hinge loss function h/rho
        y = x.*(x>1) + (1-1/rho<=x).*(x<=1) + (x+1/rho).*(x<1-1/rho);
    end

    function y = hinge(x)
        y = max(1-x,0);
    end
end