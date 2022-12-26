function RecMatTrap = GenerateRecMatTrap(paraFPPA)
%% generate the transposition of reconstruction matrix A
WaveName = paraFPPA.WaveName;
RecLev = paraFPPA.RecLev; 
N = RecLev(end);
Mat = zeros(N,N);                                                              
dwtmode('per')      % peroidic condition to extend signal
for i=1:1:N
    a=zeros(N,1); a(i)=1;
    Mat(:,i) = waverec(a,RecLev,WaveName);
end

RecMatTrap = Mat';

end