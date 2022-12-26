function [w1, v1,TargetValue] = EpsilonInsensitiveLoss_FPPA(A,b,paraFPPA)
%---------------------------------------------------------------------------
%    min phi(w) + psi(Aw), w = [alpha b]
%    For regression with epsilon-insensitive loss, 
%    phi is regularization term,  phi(x) = lambda*|x|,
%    psi is epsilon-insensitive hinge loss,   psi_b(x) = max(|x-b|-epsilon,0)
%---------------------------------------------------------------------------

lambda = paraFPPA.lambda;
MaxIter = paraFPPA.MaxIter;
rho = paraFPPA.rho;
beta = paraFPPA.beta;

epsilon = paraFPPA.epsilon; 

n=size(A,2); 
w1 = zeros(n,1);
v1 = zeros(n-1,1);

TargetValue = zeros(1,MaxIter);
%   FPPA Iteration
for k = 1:1:MaxIter
    % w update
    w2 = prox_abs(w1-beta*A.'*v1, beta*lambda);
    % v update
    temp = 1/rho*v1 + A*(2*w2-w1);
    v2 = rho*(temp-prox_epsilon_insensitive_loss(temp,rho,epsilon,b));
    
    w1 = w2;
    v1 = v2;
    
    % record
    TargetValue(k) = lambda*norm(w1(1:end-1),1) + sum(epsilon_insensitive_loss(A*w1,epsilon,b));  % Function value
    if rem(k,1000)==0
        fprintf('Iter: %d, TargetValue: %f;  \n', k, TargetValue(k))
    end
end

    function y = prox_abs(x,tau)
        %% prox operator for absolute value function $tau*| |$
        y = sign(x).*max(abs(x)-tau,0);
        y(end) = x(end);
    end

    function y = prox_epsilon_insensitive_loss(x,rho,epsilon,b)
        %% prox operator for epsilon-insensitive hinge loss function "psi_{b,epsilon}/rho"
        rho1 = 1/rho;
        if epsilon>=rho1/2
            y = (x - rho1).*(x>=epsilon+rho1+b) + (epsilon+b).*(epsilon+b<=x).*(x<epsilon+rho1+b) + x.*(epsilon-rho1+b<=x).*(x<epsilon+b)...
                + (x+rho1).*(-epsilon+b<=x).*(x<epsilon-rho1+b) + (-epsilon+b).*(-epsilon-rho1+b<=x).*(x<-epsilon+b)...
                + (x+rho1).*(x<-epsilon-rho1+b);
        else
            y = (x - rho1).*(x>=epsilon+rho1+b) + (epsilon+b).*(epsilon+b<=x).*(x<epsilon+rho1+b) + x.*(-epsilon+b<=x).*(x<epsilon+b)...
                + (-epsilon+b).*(-epsilon-rho1+b<=x).*(x<-epsilon+b) + (x+rho1).*(x<-epsilon-rho1+b);
            
        end
        
    end

    function y = epsilon_insensitive_loss(x,epsilon,b)
        %% epsilon-insensitive hinge loss
           % psi_{b,epsilon}(x) = max(|x-b|-epsilon,0)
        y = max(abs(x-b)-epsilon,0);
    end
end