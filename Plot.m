clear
clc
close all
warning off;
LowRankRate = 0.05;
res = zeros(3 , 6);
path = './';
addpath(genpath(path));

load GBM.mat
Data{1,1} = ProgressData(Gene);
Data{1,2} = ProgressData(Methy);
Data{1,3} = ProgressData(Mirna);
label = table2array(Response(:,2:end));
KH(:,:,1)=kernel_matrix(Data{1,1},'RBF_kernel',2^(15));
KH(:,:,2)=kernel_matrix(Data{1,2},'RBF_kernel',2^(9));
KH(:,:,3)=kernel_matrix(Data{1,3},'RBF_kernel',2^(17));

Y = table2array(Response(:,2:end));
z=[10;104;140;183;186;194];

    num_views=3;
    numclass =6;
    KH = kcenter(KH);
    KH = knorm(KH);
    [KHL] = LocalKernelCalculation(KH  , 0.05);
    LargestIteration = 15;
    KernelWeight = ones(1 , num_views) / num_views;
    avgKer=sumKbeta(KHL,KernelWeight);
    BaseValue = norm(avgKer , 'fro')^2;
    RegularizationValue = 10^(-4) * BaseValue;
    %Alpha_T = 0 * BaseValue;
    %Beta_T = 2.^[-8 , -2 , 2 , 6];
    NNRate = [0.01 , 0.03 , 0.09 , 0.11];
    lanbad =0.1;
    SampleNum = size(KH,1);
    LowRankRate = 0.05;
    for IRate = 1 : length(NNRate)
        CRate = NNRate(IRate);
        [KHL , M1] = LGMKCPreprocessFinal(KH , CRate);

        % for IBeta = 1 : length(Beta_T)
              %Beta_C = Beta_T(IBeta);
              InputKnum = max(2* numclass , round(LowRankRate * SampleNum) );
              [Mu , Z , flag , TotalObj] = LGMKCNew1(KHL, M1, RegularizationValue ,lanbad, zeros(SampleNum) , LargestIteration , InputKnum);
              PON = TotalObj(1:end-1) - TotalObj(2 : end);
              DON = sum(PON < -10^(-10)) == 0;
              PI = Z > 0;
              Z = Z.*PI;
              [U7] = baseline_spectral_onkernel( abs( (Z + Z') / 2) , numclass);
              U_normalized=U7./ repmat(sqrt(sum(U7.^2, 2)), 1,numclass);
              indx = litekmeans(U_normalized,numclass, 'MaxIter',100, 'Start',z,'Replicates',30);
              group = num2str(indx);
              group1 = num2cell(group);
              [p] = MatSurv(Y(:,1),Y(:,2),group1,'CensorLineLength',0,'LineWidth',1.2,'CensorLineWidth',1);
              
        % end
    end
