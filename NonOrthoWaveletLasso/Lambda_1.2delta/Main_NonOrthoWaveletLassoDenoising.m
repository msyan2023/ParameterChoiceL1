%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Parameter choices lambda=1.2*delta balancing sparsity 
%  level SL and accuracy ERR for signal denoising
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc, clear, close all
dbstop if error

Lev = 12;
N = 2^Lev;

%% Initialize signal
x = linspace(0,1,N);
Doppler = @(t) sqrt(t.*(1-t)).*sin(2*pi*1.05./(t + 0.05)); 
signal = Doppler(x)';

%% Add noise
rng(1)
SNR = 118;     % SNR = 118, 100, 85, 60, 30, 20, 10
signal_noi = awgn(signal,SNR,'measured');  % add Gaussian white noise to signal
delta = norm(signal_noi-signal,2)

dwtmode('per')    % set up discrete wavelet transform with periodic boundary condition (in this way, wavelet coefficients keep the same dimension with signal)
paraFPPA.WaveName = 'bior2.2';
paraFPPA.DecLev = 6;   % decomposion level  (Lev-DecLev is the most coarst level)


%%  Wavelet decomposition
[mniWaveletCoeff,paraFPPA.RecLev] = wavedec(signal,paraFPPA.DecLev,paraFPPA.WaveName);
% mniWaveletCoeff vector is the unique solution of uncorrupted minimal norm interpolation problem

%% Parameter choice
paraFPPA.lambda = 1.2*delta;

%% Generate matrix A
RecMatTrap = GenerateRecMatTrap(paraFPPA);    % RecMatTrap = A';
RecMat = RecMatTrap';

%% Numerical solution by FPPA 
paraFPPA.MaxIter = 1000;
paraFPPA.rho = 0.01;
paraFPPA.beta = 1/(norm(RecMat,2)^2)/paraFPPA.rho*0.999;  % convergence condition
[WaveletCoeff,TargetValue] = NonOrthoWaveletLasso_FPPA(signal_noi,RecMatTrap,paraFPPA);

%%  Wavelet reconstruction
RecSignal = waverec(WaveletCoeff,paraFPPA.RecLev,paraFPPA.WaveName); % reconstructed signal with numerical solution(wavelet coefficients)
SL = nnz(WaveletCoeff);
ERR = norm(WaveletCoeff-mniWaveletCoeff,2);
ERR_delta = ERR/delta;

lambda = paraFPPA.lambda;
fname = sprintf('Result_delta%.4e_SL_%d_ERR%d.mat',delta,SL,ERR);
save(fname , 'SL' ,'paraFPPA', 'delta', 'SNR', 'ERR', 'ERR_delta', 'Lev' )