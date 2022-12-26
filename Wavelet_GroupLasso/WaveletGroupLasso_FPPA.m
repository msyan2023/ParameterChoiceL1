function [u0,TargetValue] = WaveletGroupLasso_FPPA(c,paraFPPA)
% Solves the following group lasso problem via SingleStep FPPA:
%
%    minimize 1/2*|| Au - c ||_2^2 + lambda * \sum_i  delta_i * || u_i ||_2
%    u = [u1 u2 ...]
%    A is wavelet matrix 

lambda = paraFPPA.lambda;
rho = paraFPPA.rho;
beta = paraFPPA.beta;
MaxIter = paraFPPA.MaxIter;

WaveName = paraFPPA.WaveName;
DecLev = paraFPPA.DecLev;

delta = paraFPPA.delta;      %  regularization parameter for each group
group_info = paraFPPA.group_info;     % starting index of each group

u0 = zeros(length(c),1);
z0 = zeros(length(c),1);

TargetValue = zeros(1,MaxIter);
% FPPA Iteration
for k = 1:MaxIter
    % u update
    [A_top_z0,L] = wavedec(z0,DecLev,WaveName);    % A'*z0
    u1 = prox_group_l2(u0-beta*A_top_z0, beta*lambda*delta, group_info);

    % z update
    temp = 1/rho*z0 + waverec(2*u1-u0,L,WaveName);
    z1 = rho*(temp-prox_square(temp,rho,c));
    
    u0 = u1;
    z0 = z1;
    
    group_norm = 0;
    for j=1:1:length(group_info)-1
        group_norm = group_norm + delta(j)*norm(u0(group_info(j):group_info(j+1)-1),2);
    end
    
    TargetValue(k) = 1/2*sum((waverec(u0,L,WaveName)-c).^2) + lambda*group_norm;  % Function value
    if rem(k,100)==0
        fprintf('Iter: %d, TargetValue: %f;  \n', k, TargetValue(k))
    end
    
end
disp(' ')
    function y = prox_group_l2(x,tau,group_info)
       %% prox operator for absolute value function $tau1*||group1||_2+tau2*||group2||_2+...+tau_n*||group_n||_2$
        % "group_info" stores the starting index of each group subvector of "x"
        y = zeros(length(x),1);
        for i=1:1:length(group_info)-1
            x_sub = x(group_info(i):group_info(i+1)-1);
            if norm(x_sub,2)-tau(i)>0
                y(group_info(i):group_info(i+1)-1) = (norm(x_sub,2)-tau(i))/norm(x_sub,2)*x_sub;
            else 
                y(group_info(i):group_info(i+1)-1) = 0;
            end
        end
    end

    function y = prox_square(x,rho,c)
       %% prox operator for square function $1/rho*1/2*||variable-c||_2^2$
        rho1 = 1/rho;
        y = (rho1*c+x)/(rho1+1);
    end

end

