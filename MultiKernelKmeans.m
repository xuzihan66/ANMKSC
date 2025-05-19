%% ---- Multiple Kernel K-Means(MKKM)----- %%
%tic;
%[H_normalized0,Sigma0,obj0] = mkkmeans_train(KH,numclass);
%[res_mean0,res_std0] = myNMIACCV2(H_normalized0,Y,numclass);
%U_normalized = H_normalized0 ./ repmat(sqrt(sum(H_normalized0.^2, 2)), 1,numclass);
%indx = litekmeans(U_normalized,numclass, 'MaxIter',100, 'Replicates',10);
%[SHI,h]=silhouette(U_normalized,indx);
%SHI0=sum(SHI)/num;
%[p,fh,stats] = MatSurv(label(:,1),label(:,2),indx)
%timecost0 = toc;
function [acc] = MultiKernelKmeans(KH, Y,numclass,z)
  
   [H_normalized0,Sigma0,obj0] = mkkmeans_train(KH,numclass);
   U_normalized = H_normalized0 ./ repmat(sqrt(sum(H_normalized0.^2, 2)), 1,numclass);
   indx = litekmeans(U_normalized,numclass, 'MaxIter',100,'Start',z, 'Replicates',10);
   indx = indx(:);
   [newIndx] = bestMap(Y,indx);
   acc = mean(Y==newIndx);
   
       
end