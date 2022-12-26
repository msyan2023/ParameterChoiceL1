% Generate first order difference matrix B, its sigular value 
% decomposition and B_prime in the paper

%% Compute singular value decomposition of B
B = -1*speye(NumSamples) + sparse((1:NumSamples),[(2:NumSamples) 1],1);
B(end,:) = [];
B = full(B); 
[U,S,V] = svd(B);
save('MatB_and_SVD','U','S','V','B')

%% Compute B_prime in the paper 
s=sum(S,1); s(end)=1;
S_prime = diag(s.^(-1));
U_prime =[U' zeros(NumSamples-1,1);zeros(1,NumSamples-1) 1];
B_prime = V*S_prime*U_prime;
save('MatB_prime','B_prime')