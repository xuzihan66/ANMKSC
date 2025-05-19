function kernels = generate_gaussian_kernels(data,num_kernels,num_views)
    kernels = {};

    for i = 1:num_views
        for j = 1:num_kernels
       
            kernels{i,j}=kernel_matrix(data{1,i},'RBF_kernel', 2^(j));
        end
    end
end
