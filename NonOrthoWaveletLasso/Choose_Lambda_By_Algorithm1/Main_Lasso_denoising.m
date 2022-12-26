%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Parameter choices lambda^* balancing sparsity level l^* and accuracy ERR for   
%  signal denoising (noise delta = 0.0190, initial parameter lambda^0 = 3.9478 
%  of wavelet coefficients n := 4096)
%
%  Model:  Signal denoising model by a biorthogonal wavelet transform
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear, clc,close all
disp('Signal denoising model by a biorthogonal wavelet transform ')
disp('----------------------------------------------')
format short
dbstop if error

Lev = 12;
N = 2^Lev;

%% Initialize signal 
x = linspace(0,1,N);
Doppler = @(t) sqrt(t.*(1-t)).*sin(2*pi*1.05./(t + 0.05)); 
signal = Doppler(x)';   % original signal


%% Add noise
rng(1)
SNR = 60;
signal_noi = awgn(signal,SNR,'measured');    % add Gaussian white noise to signal
delta = norm(signal_noi-signal,2);    % delta = 0.0190

dwtmode('per')     % set up discrete wavelet transform with periodic boundary condition (in this way, wavelet coefficients keep the same dimension with signal)
paraFPPA.WaveName = 'bior2.2';
paraFPPA.DecLev = 6;     % decomposion level  (Lev-DecLev is the most coarst level)


%% Wavelet decomposition
[mniWaveletCoeff,paraFPPA.RecLev] = wavedec(signal,paraFPPA.DecLev,paraFPPA.WaveName);
% mniWaveletCoeff vector is the unique solution of uncorrupted minimal norm interpolation problem

%% Generate matrix A
RecMatTrap = GenerateRecMatTrap(paraFPPA);      % RecMatTrap = A';
RecMat = RecMatTrap';   

%% Choose parameter in FPPA algorithm
paraFPPA.TargetNNZ = 600;     %  targeted sparsity levels
paraFPPA.MaxIter = 1000;
paraFPPA.rho = 0.01;
paraFPPA.beta = 1/(norm(RecMat,2)^2)/paraFPPA.rho*0.999;  % convergence condition

%% Choose initial lambda
Lambda_max = max(abs(RecMat'*(signal_noi)));
paraFPPA.lambda = Lambda_max;
fname = sprintf('TargetNNZ%d_Tag0.mat',paraFPPA.TargetNNZ);
[SOLUTION, ~] = NonOrthoWaveletLasso_FPPA(signal_noi,RecMatTrap,paraFPPA);
Result.NNZ = nnz(SOLUTION);
save(fname, 'SOLUTION', 'Result', 'paraFPPA')

%% Solution for subsequent experiments
NumExp = 100;        % number of experiments
for LastTag = 0:1:NumExp      % we shall select lambda based on the solution obtained in the last step   
    if abs(Result.NNZ - paraFPPA.TargetNNZ)==0
        break
    end
    fprintf('\n-------------------- Tag=%d -------------------\n',LastTag+1)
    load(['TargetNNZ' num2str(paraFPPA.TargetNNZ, '%d') '_Tag' num2str(LastTag) '.mat'])
    if Result.NNZ < paraFPPA.TargetNNZ
        %% if the nnz of previous solution is smaller than target, update RHS
        RHS = sort(abs(RecMat'*(RecMat*SOLUTION - signal_noi)));
        a = max(RHS(RHS<paraFPPA.lambda));
        paraFPPA.lambda = min(RHS(N-paraFPPA.TargetNNZ),a);   % regularization parameter
        if isfield(paraFPPA,'indexRHS_GETlambda')
            paraFPPA = rmfield(paraFPPA,'indexRHS_GETlambda');
        end
    else 
        %% if the nnz of previous solution is bigger than target, we will use the previous RHS to select lambda
        if ~isfield(paraFPPA,'indexRHS_GETlambda')
            % if this is the first time choosing lambda from the solution whose nnz is bigger than target
            paraFPPA.indexRHS_GETlambda = paraFPPA.TargetNNZ - (Result.NNZ - paraFPPA.TargetNNZ);
        else
            % if this is not the first time, update the index based on the last index
            paraFPPA.indexRHS_GETlambda = paraFPPA.indexRHS_GETlambda - (Result.NNZ - paraFPPA.TargetNNZ);
        end
        paraFPPA.lambda = min(RHS(N - paraFPPA.indexRHS_GETlambda),a);   % regularization parameter
    end
    
    %------------------------------------------------------------------
    paraFPPA.Tag = LastTag + 1;   % current Tag
    
    paraFPPA
    %------------------------------------------------------------------
    
    %% Training
    [SOLUTION, ~] =  NonOrthoWaveletLasso_FPPA(signal_noi,RecMatTrap,paraFPPA);
    fname = sprintf('TargetNNZ%d_Tag%d.mat',paraFPPA.TargetNNZ,paraFPPA.Tag)
    
    %% Show accuracy
    RecSignal = waverec(SOLUTION,paraFPPA.RecLev,paraFPPA.WaveName); % reconstructed signal with numerical solution(wavelet coefficients)
    Result.ERR = norm(SOLUTION-mniWaveletCoeff,2);
    Result.NNZ = nnz(SOLUTION);
    disp('****** Results ******')
    fprintf('Number of Nonzeros of solution with length %d is <strong>%d</strong>\n', N, nnz(SOLUTION))
    fprintf('lambda_star = %f\n', paraFPPA.lambda)
    fprintf('ERR = %f\n', Result.ERR)
    fprintf('SL = %d\n', Result.NNZ)
    
    save(fname, 'SOLUTION', 'Result', 'paraFPPA', 'RHS')
  
end

