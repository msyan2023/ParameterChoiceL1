%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  (data source: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/)
%   Verify characterizations of the solutions of the l_1 SVM model for 
%   regression, the benchmark dataset is¡®Mg¡¯. 
%   
%   Model: Regression with the epsilon-insensitive loss function
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear, clc, close all
disp('Regression-EpsilonInsensitive Loss-FPPA')
disp('----------------------------------------------')
format short
dbstop if error
ReadData

%% Kernel Parameter
sigma = 1.5;   % Gaussian kernel parameter

%% Generate Kernel Matrix 
KernelMat = GaussKernel(TrainData, TrainData, sigma);
KernelMatEXT = [KernelMat ones(NumTrain,1)];

%% Generate Prediction Matrix
PredMat = GaussKernel(TrainData,TestData, sigma);
PredMatEXT = [PredMat ones(size(PredMat,1),1)];

%% Choose parameter in FPPA algorithm
paraFPPA.lambda = 1;       % regularization parameter
paraFPPA.MaxIter = 500000;      % maximal iteration in FPPA; If the parameter is small,  we need to set more steps to make it converge                                
paraFPPA.rho = 0.01; 
paraFPPA.beta = 1/(norm(KernelMatEXT,2)^2)/paraFPPA.rho*0.999;    % convergence condition 
paraFPPA.epsilon = 1e-4;  


%%  Training
[w, v,TargetValue] = EpsilonInsensitiveLoss_FPPA(KernelMatEXT, TrainLabel,paraFPPA);

%%  Plot the curve of target value
disp(' ')
figure, plot(1:1:paraFPPA.MaxIter, TargetValue), title('Target Function Value (Regression-EpsionInsenstive-L1-FPPA))'), xlabel('Iter')    

%% Show accuracy
fprintf('Number of Nonzeros of alpha: %d\n', nnz(w(1:end-1)))
ShowAccuracyRegression(KernelMatEXT*w, TrainLabel, 'Training');   
ShowAccuracyRegression(PredMatEXT*w, TestLabel, 'Testing');   

%% Check sparsity Theory
NumericalErrorTOL = 5e-5;
Gamma = CheckSparsityTheory_NonSmooth_EpsilonInsensitiveLoss(w, KernelMatEXT, paraFPPA.epsilon, TrainLabel, paraFPPA.lambda, NumericalErrorTOL);

%% Check enough iteration or not
[ValueMin, IndexMin] = min(TargetValue);
fprintf('\nThe minimum Target Value achieves at <strong>%d</strong> iteration\n', IndexMin)
fprintf('<strong>If it is too closed to the Maximum iteration %d, it is better to iterate more!!!</strong>\n', paraFPPA.MaxIter)
