
%% Compute Lipschitz constant
A = B_prime(:,1:end-1);
Lipschitz_2 = sqrt(diag(A'*A));


%% Check relaxed condition 
Condition = sort(abs(B_prime(:,1:end-1).'*(v-signal_noi))+paraFPPA.EPS*Lipschitz_2);
ESL = size(B,1)-sum(paraFPPA.lambda>Condition)    % we denote the integer l determined by proposition 4.15 by ESL





