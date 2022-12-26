%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%       Given a target nnz(number of nonzeros), this is an iteration scheme
%       of choosing regularization parameter "lambda" to make the corresponding
%       solution having prescribed nnz
%
%       Model:  Regression with square loss
%       (data source: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear, clc,close all
disp('Regression-Square Loss-FPPA')
disp('----------------------------------------------')

format short
dbstop if error

[TrainData, TestData, TrainLabel, TestLabel] = ReadData('Mg');
NumTrain = length(TrainLabel);
NumTest = length(TestLabel);
%% Kernel Parameter
paraFPPA.sigma = 1.07;   % Gaussian kernel parameter
%% Generate Kernel Matrix
KernelMat = GaussKernel(TrainData, TrainData, paraFPPA.sigma);
KernelMatEXT = [KernelMat ones(NumTrain,1)];

%% Generate Prediction Matrix
PredMat = GaussKernel(TrainData,TestData, paraFPPA.sigma);
PredMatEXT = [PredMat ones(NumTest,1)];

%% Choose parameter in FPPA algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
paraFPPA.TargetNNZ = 500;   %  targeted sparsity levels
paraFPPA.MaxIter = 5e4;
paraFPPA.rho = 0.002;
paraFPPA.beta = 1/(norm(KernelMatEXT,2)^2)/paraFPPA.rho*0.999;  % convergence condition

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%     rule for name of output: REGression SQuare loss
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Choose relaxed threhold
paraFPPA.Thres = 0;   % tolerance between nnz of solutions and target nnz

%% Choose initial lambda
paraFPPA.Initial_lambda = 0.03;  % Set a positive number to "paraFPPA.Initial_lambda" as an initial guess for lambda

fname = sprintf('REG-SQ-TargetNNZ%d_Tag0.mat',paraFPPA.TargetNNZ);
paraFPPA.lambda = paraFPPA.Initial_lambda;
paraFPPA
[SOLUTION, ~] = LassoSVM_FPPA(KernelMatEXT,TrainLabel,paraFPPA);
Result.NNZ = nnz(SOLUTION(1:end-1));
save(fname, 'SOLUTION', 'Result', 'paraFPPA')

%% Solution for subsequent experiments
NumExp = 100;    % number of experiments
for LastTag=0:1:NumExp    %  we shall select lambda based on the solution obtained in the last step
    fprintf('\n-------------------- Tag=%d -------------------\n',LastTag+1)
    load(['REG-SQ-TargetNNZ' num2str(paraFPPA.TargetNNZ, '%d') '_Tag' num2str(LastTag) '.mat'])
    if Result.NNZ < paraFPPA.TargetNNZ
        %% if the nnz of previous solution is smaller than target, update RHS
        RHS = sort(abs(KernelMat*(KernelMatEXT*SOLUTION - TrainLabel)));
        a = max(RHS(RHS<paraFPPA.lambda));
        paraFPPA.lambda = min(RHS(NumTrain-paraFPPA.TargetNNZ),a);  % regularization parameter
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
        paraFPPA.lambda = min(RHS(NumTrain - paraFPPA.indexRHS_GETlambda),a);  % regularization parameter
    end
    
    %---------------------------------------------------------
    paraFPPA.Tag = LastTag + 1;   % current Tag
    
    paraFPPA
    %---------------------------------------------------------
    
    %%  Training
    [SOLUTION, ~] = LassoSVM_FPPA(KernelMatEXT,TrainLabel,paraFPPA);
    fname = sprintf('REG-SQ-TargetNNZ%d_Tag%d.mat',paraFPPA.TargetNNZ,paraFPPA.Tag)
    %%  Show accuracy
    disp('****** Results ******')
    fprintf('Number of Nonzeros of solution with length %d is <strong>%d</strong>\n', NumTrain, nnz(SOLUTION(1:end-1)))
    Result.TrainMSE = ShowAccuracyRegression(KernelMatEXT*SOLUTION, TrainLabel, 'Training');
    Result.TestMSE = ShowAccuracyRegression(PredMatEXT*SOLUTION, TestLabel, 'Testing');
    Result.NNZ = nnz(SOLUTION(1:end-1));
    
    save(fname, 'SOLUTION', 'Result', 'paraFPPA', 'RHS')
    
    if abs(Result.NNZ - paraFPPA.TargetNNZ)<=paraFPPA.Thres
        break
    end
    
end

