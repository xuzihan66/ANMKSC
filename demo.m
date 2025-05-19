clear
clc
close all
warning off;

path = './';
addpath(genpath(path));
load LSCC.mat
data{1,1} = ProgressData(Gene);
data{1,2} = ProgressData(Methy);
data{1,3} = ProgressData(Mirna);
Y = table2array(Response(:,2:end));
num_kernels=20;

[numm,num_views]=size(data);
Kernels = generate_gaussian_kernels(data,num_kernels,num_views);
z=[37;51;75;76];
numclass =4;
[final_Kernel_selection, all_particle_positions, all_global_best_positions, all_global_best_fitness, iteration,time] = AOW(Kernels, numclass,Y, z,num_views,1, 0.9, 2,2);
