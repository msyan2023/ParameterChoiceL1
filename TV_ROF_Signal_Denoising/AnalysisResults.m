%%%%%  get all the results %%%%%%%%%%%%
clear,clc,close all

TargetNNZ = 1488; SHOW=1;

dirName = sprintf('Results-DENOI-TV-TargetNNZ%d',TargetNNZ);

mkdir(dirName),movefile("*.mat",dirName);

addpath(dirName)
a = dir([dirName '/*.mat']);
for i=1:size(a,1)-1
    load(['DENOI-TV-TargetNNZ' num2str(TargetNNZ) '_Tag' num2str(i) '.mat'])
    NNZ_set(i) = Result.NNZ;
    MSE_set(i) = Result.MSE;
    lambda_set(i) = paraFPPA.lambda;
    
    if SHOW
        fprintf('-----------------------------------%d------------------------------------', i)
        fprintf('\n lambda = %f\n', paraFPPA.lambda)
        Result
    end
end

file0 = sprintf('DENOI-TV-TargetNNZ%d_Tag0',TargetNNZ);
load(file0)
NNZ_set = [Result.NNZ NNZ_set];
lambda_set = [paraFPPA.lambda lambda_set];

figure,
p1 = plot(lambda_set, NNZ_set,'-o'); hold on;
% p2 = line(TargetNNZ, '--r','Target NNZ', 'LabelHorizontalAlignment','left');
ylim = get(gca,'Xlim');  
hold on  
plot(xlim,[TargetNNZ,TargetNNZ],'--r')
xlabel('Parameter \lambda'), ylabel('Sparsity level') 

indexTargetNNZ = find(NNZ_set==TargetNNZ); 
if ~isempty(indexTargetNNZ)
    p3 = plot(lambda_set(indexTargetNNZ),NNZ_set(indexTargetNNZ),'k*','MarkerSize',8);
    leg = legend(p3,[' Achieve targeted sparsity level ' num2str(TargetNNZ)]); 
    set(leg,'FontSize',14,'Location', 'northeast')
end



%% TargetNNZ = 355
% axis([0.5 0.552 342 357]);
% set(gca,'YTick',342:1:357)

%% TargetNNZ = 744
% axis([0.198 0.282 534 786]);
% set(gca,'YTick',534:21:786)

saveas(gcf,['DENOI-TV-TargetNNZ' num2str(paraFPPA.TargetNNZ)],'epsc')
