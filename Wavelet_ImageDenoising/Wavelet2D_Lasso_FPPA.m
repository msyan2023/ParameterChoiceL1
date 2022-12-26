function [u0,TargetValue] = Wavelet2D_Lasso_FPPA(img,paraFPPA)
% Solves the following lasso problem via SingleStep FPPA:
%
%   minimize 1/2*|| Au - c ||_2^2 + lambda* || u ||_1
%
%   A is 2-dimensional wavelet matrix 

lambda = paraFPPA.lambda;
rho = paraFPPA.rho;
beta = paraFPPA.beta;
MaxIter = paraFPPA.MaxIter;

WaveName = paraFPPA.WaveName;
DecLev = paraFPPA.DecLev;

[rows, cols] = size(img);
u0 = zeros(1,rows*cols);     % initial point 
v0 = zeros(rows,cols);

TargetValue = zeros(MaxIter,1);
% FPPA Iteration
for k = 1:MaxIter
    % u update
    [A_top_v0,L] = wavedec2(v0,DecLev,WaveName);     
    u1 = prox_abs(u0 - beta*A_top_v0,beta*lambda);
    
    % z update
    temp = 1/rho*v0 + waverec2(2*u1-u0,L,WaveName);   
    v1 = rho*(temp - prox_square(temp,rho,img));
    
    u0 = u1;
    v0 = v1;
    
    TargetValue(k) = 1/2*sum(sum((waverec2(u0,L,WaveName)-img).^2)) + lambda*norm(u0(:),1);     % Function value
    if rem(k,10)==0
        fprintf('Iter: %d, TargetValue: %f; \n ', k, TargetValue(k))
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

