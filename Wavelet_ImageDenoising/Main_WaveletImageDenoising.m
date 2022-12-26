%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%       Given a targetted sparsity levels l_star, this is a strategy 
%       of choosing regularization parameter "lambda" to make the
%       corresponding solutions enjoy the targeted sparsity levels. 
%      
%       Model: Image denoising by a wavelet transform   
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear; close all
disp('Image denoising by a wavelet transform')
disp('----------------------------------------------')

format short
Im_ref = imread('cameraman.tif');      % original image
[rows,cols] = size(Im_ref);
n = 256^2; 
Lev = 8;

%% Add noise
rng(1);
sigma = 20;          % the standard deviation of the Gaussian noise.
newsigma = (sigma^2)/(255^2);
Im_obs = imnoise(Im_ref,'gaussian',0,newsigma);       % add Gaussian white noise with mean 0 and variance newsigma.
Im_ref = double(Im_ref);       
Im_obs = double(Im_obs);    

dwtmode('per')        % peroidic condition to extend signal
paraFPPA.WaveName = 'db4';
paraFPPA.DecLev = 4;       % decomposion level  (Lev-DecLev is the most coarst level)

%% Wavelet decomposition
[WaveletCoeff,Bookkeeping] = wavedec2(Im_obs,paraFPPA.DecLev, paraFPPA.WaveName);

%% Parameter choice 
b = sort(abs(WaveletCoeff))';   
l_star = 20000;       % targetted sparsity levels 
paraFPPA.lambda = b(n-l_star);

%% Numerical solution by FPPA 
paraFPPA.rho = 1;
paraFPPA.beta = 1/paraFPPA.rho*0.99;  %% convergence condition 
paraFPPA.MaxIter = 100;
[numCoeff,TargetValue] = Wavelet2D_Lasso_FPPA(Im_obs,paraFPPA);

%% Exact solution
extCoeff = (WaveletCoeff-paraFPPA.lambda).*(WaveletCoeff-paraFPPA.lambda>0) + (WaveletCoeff+paraFPPA.lambda).*(WaveletCoeff-paraFPPA.lambda<0);
extCoeff = extCoeff.*(abs(WaveletCoeff)>paraFPPA.lambda);

%% Compare numerical solution with exact solution
NumericalError = norm(numCoeff(:) - extCoeff(:),2);

%% Wavelet reconstruction
Im_rec = waverec2(extCoeff,Bookkeeping,paraFPPA.WaveName);  % denoised image

%% PSNR values of the denoised images
[ximage,yimage] = size(Im_rec);
MSE = sum(sum((Im_ref-Im_rec).^2))/(ximage*yimage);
PSNR = 10*log10(255^2/MSE);    % PSNR = 20*log10(255*256/norm(Im_ref(:)-Im_rec(:),2));
fprintf(['Total Number of coefficients: ',num2str(rows*cols),'\n']);
fprintf(['l_star ',num2str(l_star),';',' Selected_lambda: ',num2str(paraFPPA.lambda),'\n']);
fprintf(['SL: ',num2str(nnz(extCoeff)),';', ' PSNR: ',num2str(PSNR),'\n']);

figure, imshow(uint8(Im_ref)), title('Original Image')
set(gca,'position',[0,0.02,1,1]); saveas(gcf,'Original.eps')
figure, imshow(uint8(Im_obs)), title('Noised Image')
set(gca,'position',[0,0.02,1,1]); saveas(gcf,'Noised.eps')
figure, imshow(uint8(Im_rec)), title('Denoised Image')
set(gca,'position',[0,0.02,1,1]); saveas(gcf,'Denoised.eps')