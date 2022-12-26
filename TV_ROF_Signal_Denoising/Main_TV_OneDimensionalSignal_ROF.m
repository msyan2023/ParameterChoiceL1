%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%       Given a target nnz(number of nonzeros), this is an iteration scheme
%       of choosing regularization parameter "lambda" to make the corresponding
%       solution having prescribed nnz
%
%       Model: Total-variation signal denoising
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear,clc,close all
disp('Total-variation signal denoising')
disp('----------------------------------------------')

format short
NumSamples = 4096;
% rank of first order difference matrix = n-1

%---------------------------------------------
Generate_B_and_B_prime
%---------------------------------------------

%% Initialize signal 
x = linspace(0,1,NumSamples)';
Doppler = @(t) sqrt(t.*(1-t)).*sin(2*pi*1.05./(t + 0.05));
signal = Doppler(x);   % original signal

%% Add noise
rng(1)
SNR = 7;
signal_noi = awgn(signal,SNR,'measured');  % add Gaussian white noise to signal

%% Choose parameter in FPPA algorithm
paraFPPA.TargetNNZ = 1488;   %  targeted sparsity levels
paraFPPA.MaxIter = 50000;
paraFPPA.rho = 500;
paraFPPA.beta = 1/(norm(B,2)^2)/paraFPPA.rho*0.999;  % convergence condition


%% Choose relaxed threhold
paraFPPA.Thres = 0;    % tolerance between nnz of solutions and target nnz


%% Choose initial lambda
Lambda_max = max(abs(B_prime(:,1:end-1)'*(signal_noi)));
paraFPPA.Initial_lambda = 0.2; % Set a positive number to "paraFPPA.Initial_lambda" as an initial guess for lambda
fname = sprintf('DENOI-TV-TargetNNZ%d_Tag0.mat', paraFPPA.TargetNNZ);

paraFPPA.lambda = paraFPPA.Initial_lambda;
[SOLUTION, B_SOLUTION, ~] = ROF_FPPA(B,signal_noi,paraFPPA);

Result.NNZ = nnz(B_SOLUTION);   % nnz is of the vector: FirstOrderDiffMatrix * SOLUTION
save(fname, 'SOLUTION', 'Result', 'paraFPPA')


%% Solution for subsequent experiments
NumExp = 200;    % number of experiments
for LastTag=0:1:NumExp    %  we shall select lambda based on the solution obtained in the last step
    if abs(Result.NNZ - paraFPPA.TargetNNZ) <= paraFPPA.Thres
        break
    end
    fprintf('\n-------------------- Tag=%d -------------------\n',LastTag+1)
    load(['DENOI-TV-TargetNNZ' num2str(paraFPPA.TargetNNZ, '%d') '_Tag' num2str(LastTag) '.mat'])
    if Result.NNZ < paraFPPA.TargetNNZ
        %% if the nnz of previous solution is smaller than target, update RHS
        RHS = sort(abs(B_prime(:,1:end-1).'*(SOLUTION-signal_noi)));
        a = max(RHS(RHS<paraFPPA.lambda));
        paraFPPA.lambda = min(RHS(NumSamples-1-paraFPPA.TargetNNZ),a);   % regularization parameter
        if isfield(paraFPPA, 'indexRHS_GETlambda')
            paraFPPA = rmfield(paraFPPA, 'indexRHS_GETlambda');
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
        paraFPPA.lambda =  min(RHS(NumSamples - 1 - paraFPPA.indexRHS_GETlambda),a);     % regularization parameter
    end
    
    %---------------------------------------------------------------------
    paraFPPA.Tag = LastTag + 1;   % current Tag
    
    paraFPPA
    %---------------------------------------------------------------------
    
    %%  Training
    [SOLUTION, B_SOLUTION, ~] = ROF_FPPA(B,signal_noi,paraFPPA);
    fname = sprintf('DENOI-TV-TargetNNZ%d_Tag%d.mat', paraFPPA.TargetNNZ, paraFPPA.Tag)
    %%  Show accuracy
    Result.MSE =  sum((SOLUTION - signal).^2)/NumSamples;
    Result.NNZ = nnz(B_SOLUTION);
    disp('****** Results ******')
    fprintf('lambda_star = %f\n', paraFPPA.lambda)
    fprintf('MSE = %f\n', Result.MSE)
    fprintf('SL = %d\n', Result.NNZ)
    save(fname, 'SOLUTION', 'Result', 'paraFPPA', 'RHS')
    
end








