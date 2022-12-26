%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%       Given a targetted sparsity levels l_star, this is a strategy 
%       of choosing regularization parameter "lambda" to make the
%       corresponding solutions enjoy the targeted sparsity levels. 
%      
%       Model:  Signal denoising by the group Lasso regularized model  
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear, clc,close all;
disp('Signal denoising by the group Lasso')
disp('----------------------------------------------')

Lev = 12;
N = 2^Lev;

%% Initialize signal 
x = linspace(0,1,N)';
Doppler = @(t) sqrt(t.*(1-t)).*sin(2*pi*1.05./(t + 0.05)); 
signal = Doppler(x);   % original signal

%% Add noise
rng(1);
SNR = 7;
signal_noi = awgn(signal,SNR,'measured');   %  add Gaussian white noise to signal

%%
dwtmode('per')       %  peroidic condition to extend signal
paraFPPA.WaveName = 'db6';
paraFPPA.DecLev = 9;      %  decomposion level  (Lev-DecLev is the most coarst level)


%% Wavelet decomposition
[Wx, L] = wavedec(signal_noi, paraFPPA.DecLev, paraFPPA.WaveName);     % wavelet decomposition 
AppCoeff = appcoef(Wx,L,paraFPPA.WaveName);       % extracts the low frequency coefficients
DetailCoeff = flip(detcoef(Wx,L,paraFPPA.WaveName,'cells'));     % extracts the detail coefficients


%% 
num_group = length(L) - 1;
Wx_norm = zeros(num_group,1);      % length(Wx_norm) = length(L) - 1 = number of group
paraFPPA.delta = sqrt(L(1:end-1));    % we set delta_i as square root of the number of elements in i-th group 
Wx_norm(1) = norm(AppCoeff,2)/paraFPPA.delta(1);
for j = 2:1:num_group
    Wx_norm(j) = norm(DetailCoeff{j-1},2)/paraFPPA.delta(j);
end

group_info = zeros(num_group,1);   % starting index of each group
group_info(1) = 1;
for j = 2:1:num_group
    group_info(j) = sum(L(1:j-1)) + 1;
end
group_info = [group_info; L(end)];  % length(group_info) = num_group 
                                     % The last number in group_info is the number of all coefficients
paraFPPA.group_info = group_info;


%% Parameter choice stategy 
a = sort(Wx_norm);
l_star = 0;
paraFPPA.lambda = a(length(a)- l_star);

%% Numerical solution by FPPA
paraFPPA.rho = 1;
paraFPPA.beta = 1/paraFPPA.rho*0.99;    % convergence condition
paraFPPA.MaxIter = 1000;
[Coeff,TargetValue] = WaveletGroupLasso_FPPA(signal_noi,paraFPPA);


%% Coeff grouping
Coeff_nnz = zeros(num_group,1);
for j=1:1:num_group
    Coeff_nnz(j) = nnz(Coeff(paraFPPA.group_info(j):paraFPPA.group_info(j+1)-1));
end

%% Wavelet reconstruction
RecSignal = waverec(Coeff,L,paraFPPA.WaveName);     % reconstructed signal 

%% MSE values of the denoised signal
MSE = sum((RecSignal - signal).^2)/N

fprintf(['Total number of groups: ',num2str(num_group),'\n']);
fprintf(['l_star ',num2str(l_star),';',' Selected_lambda: ',num2str(paraFPPA.lambda),'\n']);
fprintf(['BSL: ',num2str(nnz(Coeff_nnz)),';', ' MSE: ',num2str(MSE),'\n']);