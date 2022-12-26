%%%%  get all the results  %%%%%%%%%%
clc, clear, close all

TargetNNZ = 300; SHOW=1;

dirName = sprintf('Results-REG-SQ-TargetNNZ%d',TargetNNZ); 

mkdir(dirName),movefile("*.mat",dirName);

addpath(dirName)
a = dir([dirName '/*.mat']);
for i=1:size(a,1)-1
    load(['REG-SQ-TargetNNZ' num2str(TargetNNZ) '_Tag' num2str(i) '.mat'])
    NNZ_set(i) = Result.NNZ;
    TrainMSE_set(i) = Result.TrainMSE;
    TestMSE_set(i) = Result.TestMSE;
    lambda_set(i) = paraFPPA.lambda;
    
    if SHOW
        fprintf('-----------------------------------%d------------------------------------', i)
        fprintf('\n lambda = %f\n', paraFPPA.lambda)
        Result
    end
end

file0 = sprintf('REG-SQ-TargetNNZ%d_Tag0',TargetNNZ);
load(file0)
NNZ_set = [Result.NNZ NNZ_set];
lambda_set = [paraFPPA.lambda lambda_set];

figure, 
p1 = plot(lambda_set, NNZ_set,'-o'); hold on; xlabel('Parameter \lambda'), ylabel('Sparsity level')
p2 = yline(TargetNNZ, '--r', 'LabelHorizontalAlignment','left');

indexTargetNNZ = find(abs(NNZ_set-TargetNNZ)<=paraFPPA.Thres); 
if ~isempty(indexTargetNNZ)
    p3 = plot(lambda_set(indexTargetNNZ),NNZ_set(indexTargetNNZ),'k*','MarkerSize',8);
    leg = legend(p3,['Achieve Target sparsity level ',num2str(TargetNNZ),'\pm',num2str(paraFPPA.Thres)]); set(leg,'Location', 'northeast')
end
 
saveas(gcf,['REG-SQ-TargetNNZ' num2str(paraFPPA.TargetNNZ)],'epsc')