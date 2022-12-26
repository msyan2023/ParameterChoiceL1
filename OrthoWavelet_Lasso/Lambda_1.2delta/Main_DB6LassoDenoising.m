%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Parameter choices lambda=1.2*delta balancing sparsity 
%  level SL and accuracy ERR for signal denoising
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc, clear, close all

Lev = 12;
N = 2^Lev;

%% Initialize signal
x = linspace(0,1,N);
Doppler = @(t) sqrt(t.*(1-t)).*sin(2*pi*1.05./(t + 0.05)); 
signal = Doppler(x)';

%% Add noise
rng(1)
SNR = 200;   % SNR 200, 128, 100, 60, 30, 20, 10 
signal_noi = awgn(signal,SNR,'measured');  % add Gaussian white noise to signal
delta = norm(signal_noi-signal,2);


dwtmode('per')    % peroidic condition to extend signal
paraFPPA.WaveName = 'db6';
paraFPPA.DecLev = 8;  % decomposion level  (Lev-DecLev is the most coarst level)

%%  Wavelet decomposition
[mniCoeff,paraFPPA.RecLev] = wavedec(signal,paraFPPA.DecLev,paraFPPA.WaveName);
% mniCoeff vector is the unique solution of uncorrupted minimal norm interpolation problem
[z,L] = wavedec(signal_noi, paraFPPA.DecLev, paraFPPA.WaveName);


%%  Parameter choice
paraFPPA.lambda = 1.2*delta;


%% Numerical solution by FPPA 
paraFPPA.MaxIter = 1000;
paraFPPA.rho = 1;
paraFPPA.beta = 1/paraFPPA.rho*0.999;    % convergence condition
[numCoeff,TargetValue] = WaveletLasso_FPPA(signal_noi,paraFPPA);

%% Exact solution
extCoeff = (z-paraFPPA.lambda).*(z-paraFPPA.lambda>0) + (z+paraFPPA.lambda).*(z-paraFPPA.lambda<0);
extCoeff = extCoeff.*(abs(z)>paraFPPA.lambda);

%% Compare numerical solution with exact solution
NumericalError = norm(numCoeff(:) - extCoeff(:),2);

%%  Wavelet reconstruction
numRecSignal = waverec(numCoeff,L,paraFPPA.WaveName);   % reconstructed signal with numerical solution(wavelet coefficients)
extRecSignal = waverec(extCoeff,L,paraFPPA.WaveName);   % reconstructed signal with exact solution(wavelet coefficients)


SL = nnz(numCoeff);
ERR = norm(numCoeff-mniCoeff,2);
ERR_delta = ERR/delta;


lambda = paraFPPA.lambda;
fname = sprintf('Result_delta%.4e_SL-%d_ERR%0.4e.mat',delta,SL,ERR);
save(fname,'N', 'Lev', 'SL', 'paraFPPA','delta', 'SNR', 'ERR', 'ERR_delta')