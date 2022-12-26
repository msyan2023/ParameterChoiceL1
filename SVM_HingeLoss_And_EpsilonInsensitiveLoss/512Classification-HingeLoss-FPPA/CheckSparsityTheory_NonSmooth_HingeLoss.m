function Gamma = CheckSparsityTheory_NonSmooth_HingeLoss(w, MatB, lambda, NumericalErrorTOL)

IndexNonZ = find(w);  % index of nonzeros of w
IndexZ = find(~w);  % index of zeros of w

MatBw = MatB*w;  % notice that the entries of 'MatBw' are all positive


%%%%%%    subdifference of psi(Kw)  =  K.'*parital psi(Kw)        %%%%%%%%%%%%%
%%%%%%    where psi(t)=max(1-t,0)  (hinge loss)   and    K = MatB   %%%%%%%%%%%%%
ProdSubDiff = ComputeProdSubDiff(MatBw, NumericalErrorTOL, MatB);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ProdSubDiff_alpha = ProdSubDiff(1:end-1,:);
ProdSubDiff_b = ProdSubDiff(end,:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% for example   min([1 2]) = 1      min([-3 -2]) = 2          min([-1 2]) = 0
                                                                                                % 
SignProdSubDiff_alpha = sign(ProdSubDiff_alpha(:,1)).*sign(ProdSubDiff_alpha(:,2));
minProdSubDiff_alpha = min(abs(ProdSubDiff_alpha),[],2);    
minProdSubDiff_alpha(SignProdSubDiff_alpha<0) = 0;
%% check condition 1 for alpha:       -lambda*sign(u) \in ProdSubDiff_alpha
EleNonZ = -lambda*sign(w(IndexNonZ));

IntervalNonZ = ProdSubDiff(IndexNonZ,:);

TrueOrFalse = (IntervalNonZ(:,1)<=EleNonZ).*(EleNonZ<=IntervalNonZ(:,2));
disp(' ')
disp('---------------Check for condition 1 for alpha---------------')
fprintf('There are %d indices not satisfying condition 1\n\n', length(find(~TrueOrFalse)))

%% check condition 2 for alpha:       lambda > Gamma
disp('---------------Check for condition 2 for alpha---------------')
Gamma = max(minProdSubDiff_alpha(IndexZ));
fprintf('lambda=%f should be bigger than Gamma=%f\n\n', lambda, Gamma)

%% check condition 1 for b:      0 = derivative of b   (0 \in ProdSubDiff_b)
disp('---------------Check for condition 1 for b---------------')
fprintf('0 should be in [%f,%f]\n', ProdSubDiff_b(1), ProdSubDiff_b(2))

end