%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Parameter choices lambda=C*delta balancing sparsity level SL 
%  and accuracy ERR for signal denoising
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc, clear, close all

Lev = 12;
N = 2^Lev;

%% Initialize signal
x = linspace(0,1,N);
Doppler = @(t) sqrt(t.*(1-t)).*sin(2*pi*1.05./(t + 0.05)); 
signal = Doppler(x)';

%% Add noise
rng(1)
SNR = 100;    % SNR = 100, 60
signal_noi = awgn(signal,SNR,'measured');  % add Gaussian white noise to signal
delta = norm(signal_noi-signal,2);

dwtmode('per')   % set up discrete wavelet transform with periodic boundary condition (in this way, wavelet coefficients keep the same dimension with signal)
paraFPPA.WaveName = 'bior2.2';
paraFPPA.DecLev = 6;   % decomposion level  (Lev-DecLev is the most coarst level)

%% Wavelet decomposition
[mniWaveletCoeff,paraFPPA.RecLev] = wavedec(signal,paraFPPA.DecLev,paraFPPA.WaveName);
% mniWaveletCoeff vector is the unique solution of uncorrupted minimal norm interpolation problem


%% Parameter choice
C = 0.12 
paraFPPA.lambda = C*delta;

%% Numerical solution by FPPA
paraFPPA.MaxIter = 1000;
paraFPPA.rho = 0.01;
RecMatTrap = GenerateRecMatTrap(paraFPPA); % RecMatTrap = A';
RecMat = RecMatTrap';
paraFPPA.beta = 1/(norm(RecMat,2)^2)/paraFPPA.rho*0.999; % convergence condition
[WaveletCoeff,TargetValue] = NonOrthoWaveletLasso_FPPA(signal_noi,RecMatTrap,paraFPPA);

%% Wavelet reconstruction
RecSignal = waverec(WaveletCoeff,paraFPPA.RecLev,paraFPPA.WaveName);  % reconstructed signal with numerical solution(wavelet coefficients)
SL = nnz(WaveletCoeff)
ERR = norm(WaveletCoeff-mniWaveletCoeff,2);

lambda = paraFPPA.lambda;
fname = sprintf('Result_2C%4.2f_delta%.2e_SL-%d_ERR%0.4e.mat',C,delta,SL,ERR);
save(fname , 'SL','paraFPPA', 'delta', 'SNR', 'ERR', 'Lev' )
