%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   (data source: http://ufldl.stanford.edu/wiki/index.php/Using_the_MNIST_Dataset)
%    Verify characterizations of the solutions of the l_1 SVM model for classification.
%    The dataset we use is MNIST database with digits ¡®7¡¯ and ¡®9¡¯.
%
%    Model:  Classification with hinge loss (512 training dataset) 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc, clear, close all
restoredefaultpath 
addpath('Data')
disp('Classification-Hinge Loss-FPPA')
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

ImgsTrain = ImgsTrain(:,1:1:512);
LabelsTrain = LabelsTrain(1:512);

NumTrain = length(LabelsTrain);
NumTest = length(LabelsTest);

%% Kernel Parameter
sigma = 4;  % Gaussian kernel parameter

%% Generate Kernel Matrix 
KernelMat = GaussKernel(ImgsTrain, ImgsTrain, sigma);
KernelMatEXT = [KernelMat ones(NumTrain,1)];
MatB = diag(LabelsTrain)*KernelMatEXT;

%% Generate Prediction Matrix
PredMat = GaussKernel(ImgsTrain,ImgsTest, sigma);
PredMatEXT = [PredMat ones(NumTest,1)];

%% Choose parameter in proximity algorithm
paraFPPA.MaxIter = 5000000;     % maximal iteration in FPPA   
paraFPPA.lambda = 0.2;        % regularization parameter
paraFPPA.rho = 2e-05; 
paraFPPA.beta =1/(norm(MatB,2)^2)/paraFPPA.rho;    % convergence condition 

%%  Training
fname = sprintf('result0_data512_lambda%f.mat', paraFPPA.lambda)
[w, v, TargetValue] = HingeLoss_FPPA(MatB, paraFPPA); 
save(fname, 'w', 'v', 'TargetValue', 'paraFPPA')

%%  Plot the curve of target value
disp(' ')
figure, plot(1:1:paraFPPA.MaxIter, TargetValue), title('Target Function Value (SVM-HingeLoss-L1-FPPA'), xlabel('Iter')    

%% Show accuracy
fprintf('Number of Nonzeros of alpha: %d\n', nnz(w(1:end-1)))
ShowAccuracyClassification(sign(KernelMatEXT*w), LabelsTrain, 'Training')
ShowAccuracyClassification(sign(PredMatEXT*w), LabelsTest, 'Testing')

%%  Check sparsity Theory
NumericalErrorTOL = 5e-5;
Gamma = CheckSparsityTheory_NonSmooth_HingeLoss(w, MatB, paraFPPA.lambda, NumericalErrorTOL)

%% Check enough iteration or not
[ValueMin, IndexMin] = min(TargetValue);
fprintf('\nThe minimum Target Value achieves at <strong>%d</strong> iteration\n', IndexMin)
fprintf('<strong>If it is too closed to the Maximum iteration %d, it is better to iterate more!!!</strong>\n', paraFPPA.MaxIter)
