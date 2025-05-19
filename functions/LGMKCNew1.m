function [Mu , Z , flag , TotalObj,Iter] = LGMKCNew1(KHL , M1 ,  RegularizationValue ,lanbad ,L , LargestIteration , CluNum)

[SampleNum , ~ , KerNum] = size(KHL);

Mu = ones(KerNum , 1) / KerNum;
TotalObj = zeros(LargestIteration , 1);
Iter = 0;
flag = 0;
f1= 0 ;
f2= 1;
while Iter < LargestIteration && flag == 0
    CurrentKernel = sumKbeta(KHL , Mu);
    % Calculate Z
    KSquare = CurrentKernel*CurrentKernel;
    A = KSquare + RegularizationValue*eye(SampleNum) + f1 * L;
    AInv = eye(size(A)) / A;
    %     AInv = inv(A);
    TempN = KSquare * AInv * KSquare;
    TempN = (TempN + TempN') / 2;
    [N , D] = eigs(TempN , [] , CluNum, 'lm');
    
    Z = AInv * KSquare * (N * N');
    
    % Calcluate Mu
    M2 = M_2Calculation_New(KHL , Z);
    Mu = MuCulationNew(M1 , M2 , f2);
    f1= 1 ./ ((1 + abs(trace(Z' * Z))).^2);
    f2= 1 ./ (1 + exp(-0.1 * trace(Z' * L * Z)));
    
    Obj = ObjCalculation(Mu , M1 , M2 , RegularizationValue, f1, f2 , Z , L);
    Iter = Iter + 1;
    TotalObj(Iter) = Obj;
    
    if Iter >=2   
        if (abs( TotalObj(Iter)-TotalObj(Iter-1) ) < TotalObj(Iter) * 10^(-5))
            flag = 1;
        end
    end
end
TotalObj  = TotalObj(1 : Iter);