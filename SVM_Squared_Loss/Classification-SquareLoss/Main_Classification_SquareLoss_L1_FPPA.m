%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%       Given a target nnz(number of nonzeros), this is an iteration scheme
%       of choosing regularization parameter "lambda" to make the corresponding
%       solution having prescribed nnz
%
%       Model:  Classification with square loss
%       (data source: http://ufldl.stanford.edu/wiki/index.php/Using_the_MNIST_Dataset)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc, clear, close all
restoredefaultpath
addpath('Data')
disp('Classification-Square Loss-FPPA')
disp('----------------------------------------------')
format short
dbstop if error
%% Goal: binary classification of handwritten digit numbers: $num1$ and $num2$
num1 = 7;
num2 = 9;
fprintf('Handwritten Digit Recognization: %d and %d\n\n', num1, num2)

load ImgsTrain.mat
load LabelsTrain.mat
load ImgsTest.mat
load LabelsTest.mat

% ImgsTrain = ImgsTrain(:,1:512);
% LabelsTrain = LabelsTrain(1:512);

NumTrain = length(LabelsTrain);
NumTest = length(LabelsTest);

%% Kernel Parameter
paraFPPA.sigma = 4.8;  % Gaussian kernel parameter
%% Generate Kernel Matrix
KernelMat = GaussKernel(ImgsTrain, ImgsTrain, paraFPPA.sigma);
KernelMatEXT = [KernelMat ones(NumTrain,1)];
%% Generate Prediction Matrix
PredMat = GaussKernel(ImgsTrain,ImgsTest, paraFPPA.sigma);
PredMatEXT = [PredMat ones(NumTest,1)];
%% Choose parameter in FPPA algorithm
paraFPPA.TargetNNZ = 4000;   %  targeted sparsity levels
paraFPPA.MaxIter = 1e4;
paraFPPA.rho = 0.001;
paraFPPA.beta = 1/(norm(KernelMatEXT,2)^2)/paraFPPA.rho*0.999;  % beta==3.3880e-04 when rho==0.001


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%     rule for name of output: CLASSification SQuare loss
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Choose relaxed threhold
paraFPPA.Thres = 5;    % tolerance between nnz of solutions and target nnz

%% Choose initial lambda
paraFPPA.Initial_lambda = 5;  % Set a positive number to "paraFPPA.Initial_lambda" as an initial guess for lambda

paraFPPA.lambda = paraFPPA.Initial_lambda;
fname = sprintf('CLASS-SQ-TargetNNZ%d_Tag0.mat',paraFPPA.TargetNNZ);
paraFPPA
[SOLUTION, ~] = LassoSVM_FPPA(KernelMatEXT,LabelsTrain,paraFPPA);
Result.NNZ = nnz(SOLUTION(1:end-1));
save(fname, 'SOLUTION', 'Result', 'paraFPPA')

%% Solution for subsequent experiments
NumExp = 100;    % number of experiments
for LastTag=0:1:NumExp    %  we shall select lambda based on the solution obtained in the last step
    fprintf('\n-------------------- Tag=%d -------------------\n',LastTag+1)
    load(['CLASS-SQ-TargetNNZ' num2str(paraFPPA.TargetNNZ, '%d') '_Tag' num2str(LastTag) '.mat'])
    if Result.NNZ < paraFPPA.TargetNNZ
        %% if the nnz of previous solution is smaller than target, update RHS
        RHS = sort(abs(KernelMat*(KernelMatEXT*SOLUTION - LabelsTrain)));
        a = max(RHS(RHS<paraFPPA.lambda));
        paraFPPA.lambda = min(RHS(NumTrain-paraFPPA.TargetNNZ),a); % regularization parameter
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
        paraFPPA.lambda = min(RHS(NumTrain - paraFPPA.indexRHS_GETlambda),a); % regularization parameter
    end
    
    %-----------------------------------------------------------
    paraFPPA.Tag = LastTag + 1;   % current Tag
    
    paraFPPA
    %------------------------------------------------------------
    
    %%  Training
    [SOLUTION, ~] = LassoSVM_FPPA(KernelMatEXT, LabelsTrain, paraFPPA);
    fname = sprintf('CLASS-SQ-TargetNNZ%d_Tag%d.mat', paraFPPA.TargetNNZ, paraFPPA.Tag)
    %%  Show accuracy
    disp('****** Results ******')
    fprintf('Number of Nonzeros of solution with length %d is <strong>%d</strong>\n', NumTrain, nnz(SOLUTION(1:end-1)))
    Result.TrainMSE = ShowAccuracyClassification(sign(KernelMatEXT*SOLUTION), LabelsTrain, 'Training');
    Result.TestMSE = ShowAccuracyClassification(sign(PredMatEXT*SOLUTION), LabelsTest, 'Testing');
    Result.NNZ = nnz(SOLUTION(1:end-1));
    
    save(fname, 'SOLUTION', 'Result', 'paraFPPA', 'RHS')
    
    if abs(Result.NNZ - paraFPPA.TargetNNZ) <= paraFPPA.Thres
        break
    end
    
end







