%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Verify characterizations of the solutions of the 
%  Total-variation signal denoising model
%
%  Model: Total-variation signal denoising model
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc, clear, close all
disp('Total-variation signal denoising model')
disp('----------------------------------------------')

NumSamples = 4096;
% rank of first order difference matrix = n-1

%---------------------------------------------
Generate_B_and_B_prime
% load MatB_and_SVD; load MatB_prime;
%---------------------------------------------

%% Initialize signal 
x = linspace(0,1,NumSamples)';
Doppler = @(t) sqrt(t.*(1-t)).*sin(2*pi*1.05./(t + 0.05)); 
signal = Doppler(x);    % original signal

%% Add noise
rng(1)
SNR = 7;
signal_noi = awgn(signal,SNR,'measured');  % add Gaussian white noise to signal

%%  Choose seven different values of parameter lambda
Lambda_max = max(abs(B_prime(:,1:end-1)'*(signal_noi)));   
paraFPPA.lambda = 0.5;

%%  Numerical solution by FPPA algorithm
paraFPPA.MaxIter = 50000; 
paraFPPA.rho = 500;    
paraFPPA.beta = 1/(norm(B,2)^2)/paraFPPA.rho*0.999;  % convergence condition 
[u_star,B_u,~] = TVSignalDenoising_FPPA(B,signal_noi,paraFPPA);
MSE = sum((u_star - signal).^2)/NumSamples
SL = nnz(B_u)

%% Generate vector vpara
paraFPPA.EPS = 5e-5;   
v0 = rand(length(u_star),1);
v0_normalization = v0/norm(v0,2);
v = u_star + paraFPPA.EPS*v0_normalization;
Compute_eps = norm(u_star-v,2);       % satisfy the condition ||u^*-v||_2<=paraFPPA.EPS

%---------------------------------------------
RelaxedCondition;
%---------------------------------------------

lambda = paraFPPA.lambda;
fname = sprintf('Result_lambda%.4f_SL_%d_ESL%d.mat',lambda,SL,ESL);
save(fname, 'MSE' , 'SL','ESL' ,'paraFPPA', 'SNR')