function ProdSubDiff = ComputeProdSubDiff(MatBw, NumericalErrorTOL, MatB)
%%%%%%    subdifference of psi(Kw)  =  K.'*parital psi(Kw)        %%%%%%%%%%%%%
%%%%%%    where psi(t)=max(1-t,0)  (hinge loss)   and    K = MatB   %%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     Toy Example to check the program   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
% MatBw = [1.03 2 1.02].';
% NumericalErrorTOL = 0.1;
% MatB = [1 3 0; 2 2 1; -1 0 1].';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     "ProdSubDiff" will be [-1 0; -3 0; -1 1]      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

IndSubDiff_Int = find(abs(MatBw-1)<NumericalErrorTOL);  %% index of subdifference being an interval, in this case the interval is [-1,0]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SubDiff_Single = MatBw;  
SubDiff_Single(SubDiff_Single<1) = -1;
SubDiff_Single(IndSubDiff_Int) = 0;
SubDiff_Single(SubDiff_Single>1) = 0;

prodSubDiff_Single = MatB.'*SubDiff_Single;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Coeff_Int = MatB.';
Coeff_Int = Coeff_Int(:,IndSubDiff_Int);  % coeffcient of subdifference being an interval 
sumPos = sum(Coeff_Int.*(Coeff_Int>0),2);  % sum of positive coefficients
sumNeg = sum(Coeff_Int.*(Coeff_Int<0),2); % sum of negative coefficients

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ProdSubDiff = prodSubDiff_Single + [-sumPos, -sumNeg];
end