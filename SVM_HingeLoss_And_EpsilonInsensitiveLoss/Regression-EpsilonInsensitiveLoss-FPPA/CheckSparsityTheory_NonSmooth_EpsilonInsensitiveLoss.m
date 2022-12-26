function Gamma = CheckSparsityTheory_NonSmooth_EpsilonInsensitiveLoss(w, KernelMatEXT, epsilon, TrainLabel, lambda, NumericalErrorTOL)
%% w = [alpha b];
Kw = KernelMatEXT*w; 
IndSubDiff_Int_neg1 = find(abs(TrainLabel-epsilon-Kw)<NumericalErrorTOL);  % index of subdifference being [-1,0]
% fprintf('The minimum of the vector ready to truncate is %f\n', min(abs(TrainLabel-epsilon-Kw)))
IndSubDiff_Int_pos1 = find(abs(TrainLabel+epsilon-Kw)<NumericalErrorTOL);  % index of subdifference being [0,1]
% fprintf('The minimum of the vector ready to truncate is %f\n', min(abs(TrainLabel+epsilon-Kw)))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SubDiffatKw_Single = Kw;  
SubDiffatKw_Single(Kw<TrainLabel-epsilon) = -1;
SubDiffatKw_Single((TrainLabel-epsilon<Kw)&(Kw<TrainLabel+epsilon)) = 0;
SubDiffatKw_Single(TrainLabel+epsilon<Kw) = 1;

SubDiffatKw_Single(IndSubDiff_Int_neg1) = 0;
SubDiffatKw_Single(IndSubDiff_Int_pos1) = 0;

prodSubDiff_Single = KernelMatEXT.'*SubDiffatKw_Single;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Coeff_Int = KernelMatEXT.';
Coeff_Int_neg1 = Coeff_Int(:,IndSubDiff_Int_neg1);  % coeffcient of subdifference being an interval [-1,0]
Coeff_Int_pos1 = Coeff_Int(:,IndSubDiff_Int_pos1);  % coeffcient of subdifference being an interval [0,1]

sumPos_neg1 = sum(Coeff_Int_neg1.*(Coeff_Int_neg1>0),2);  % sum of positive coefficients of [-1,0]
sumNeg_neg1 = sum(Coeff_Int_neg1.*(Coeff_Int_neg1<0),2);  % sum of negative coefficients of [-1,0]

sumPos_pos1 = sum(Coeff_Int_pos1.*(Coeff_Int_pos1>0),2);  % sum of positive coefficients of [0,1]
sumNeg_pos1 = sum(Coeff_Int_pos1.*(Coeff_Int_pos1<0),2);  % sum of negative coefficients of [0,1]

%%%%%%%%%%%%%%%    subdifference of psi(Kw)  =  K.'*parital psi(Kw)        %%%%%%%%%%%%%
ProdSubDiff = prodSubDiff_Single + [-sumPos_neg1+sumNeg_pos1, -sumNeg_neg1+sumPos_pos1];
ProdSubDiff_alpha = ProdSubDiff(1:end-1,:);
ProdSubDiff_b= ProdSubDiff(end,:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

SignProdSubDiff_alpha = sign(ProdSubDiff_alpha(:,1)).*sign(ProdSubDiff_alpha(:,2));
minProdSubDiff_alpha = min(abs(ProdSubDiff_alpha),[],2);
minProdSubDiff_alpha(SignProdSubDiff_alpha<0) = 0;
%% check condition 1 for alpha:       -lambda*sign(u) \in ProdSubDiff
disp('---------------Check for condition 1 for alpha---------------')

alpha = w(1:end-1);
IndexNonZ = find(alpha);  % index of nonzeros of alpha
IndexZ = find(~alpha);  % index of zeros of alpha

EleNonZ = -lambda*sign(alpha(IndexNonZ));

IntervalNonZ = ProdSubDiff_alpha(IndexNonZ,:);

TrueOrFalse = (IntervalNonZ(:,1)<=EleNonZ).*(EleNonZ<=IntervalNonZ(:,2));

fprintf('There are %d indices not satisfying condition 1\n\n', length(find(~TrueOrFalse)))

%% check condition 2 for alpha:       lambda > Gamma
disp('---------------Check for condition 2 for alpha---------------')
Gamma = max(minProdSubDiff_alpha(IndexZ));
fprintf('lambda=%f should be bigger than Gamma=%f\n\n', lambda, Gamma)

%% check condition 1 for b:
disp('---------------Check for condition 1 for b---------------')
fprintf('0 should be in [%f,%f]\n', ProdSubDiff_b(1), ProdSubDiff_b(2))
end