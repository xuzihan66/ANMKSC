function [final_Kernel_selection, all_particle_positions, all_global_best_positions, all_global_best_fitness, iteration,time] = AOW(Kernel,numclass, Y, z,num_views,m, w, c1,c2)

%% Include dependencies
addpath('./lib'); % dependencies
addpath('./methods'); % FS methods
addpath(genpath('./lib/drtoolbox'));

tic; % start timer


% 参数设置

num_particles =40; % 粒子数量
max_iterations = 40; % �?大迭代次�?
v= num_views ; % 视图数量
% num_features_per_view = m; % 每个视图已�?�择的已排序特征数量，这里假设每个视图�?�取m个特�?
% w = 0.9; % 惯�?�权�?
% c1 = 2; % 个体学习因子
% c2 = 2; % 社会学习因子
% 初始化粒子群
particles_position = zeros(num_particles, num_views); % 初始化粒子位置，每个视图选择的特征数�?
particles_velocity = rand(num_particles, num_views); % 初始化粒子�?�度
personal_best = particles_position; % 个体�?佳位置初始化为当前位�?
global_best = []; % 全局�?佳位�?
consecutive_stagnation = 0; % 连续代数中最优个体未改变的计数器
stagnation_threshold = 5; % 设定连续代数未改变的阈�??

% 跟踪�?佳个体和全局的�?�应�?
personal_best_fitness = zeros(1, num_particles);
global_best_fitness = -inf; % 初始全局�?佳�?�应度设为负无穷

% Initialize variables to store the data
all_particle_positions = zeros(max_iterations+1, num_particles, num_views);
all_global_best_positions = zeros(max_iterations+1, num_views);

% 计算初始适应度并更新全局�?佳位�?
for g = 1:num_particles
    % 随机初始化粒子位置，确保每个视图中�?�择的特征数量合为m
    particles_position(g, :) = random_Kernel_selection(num_views, 20);
    
    % 计算适应度，包括分类准确�?
    [fitness] = compute_fitness(particles_position(g, :), num_views,numclass,Kernel,Y,z);

    % 记录个体�?佳�?�应�?
    personal_best_fitness(g) = fitness;
    
    % 更新个体�?佳位�?
    personal_best(g, :) = particles_position(g, :);
    
    % 更新全局�?佳位�?
    if  isempty(global_best) || (fitness < global_best_fitness && fitness>0 && isreal(fitness)==1)
        global_best = particles_position(g, :);
        global_best_fitness = fitness;
  
    end
end

all_particle_positions(1, :, :) = particles_position;
all_global_best_positions(1, :) = global_best;
all_global_best_fitness(1) = global_best_fitness;

previous_global_best = global_best; % 初始化前�?代的全局�?佳位�?

% �?始迭�?
for iteration = 1:max_iterations
    for g = 1:num_particles
        % 计算速度更新
        particles_velocity(g, :) = w * particles_velocity(g, :) ...
            + c1 * rand() * (personal_best(g, :) - particles_position(g, :)) ...
            + c2 * rand() * (global_best - particles_position(g, :));
        
        % 限制速度范围，确保在合理的范围内
        particles_velocity(g, :) = min(max(particles_velocity(g, :), -1), 1);
        
        % 更新粒子位置（使�? round 函数来确保整数�?�）
        particles_position(g, :) = round(particles_position(g, :) + particles_velocity(g, :));
        particles_position(g, :) = max(particles_position(g, :), 0);
        for i=1:3
            if particles_position(g,i)== 0
                particles_position(g,i)=particles_position(g,i)+1;
            elseif particles_position(g,i)== 21
                    particles_position(g,i)=particles_position(g,i)-1;
            end
            
        end
%        % 随机修正粒子位置，以确保每个视图中�?�择的特征数量合为m
%        particles_position(i, :) = random_feature_correction(particles_position(i, :), num_views, m);
        
        % 计算适应度，包括分类准确�?
        [fitness] = compute_fitness(particles_position(g, :), num_views,numclass, Kernel,Y,z);
        
        % 更新个体�?佳位�?
        if fitness < personal_best_fitness(g) && fitness>0  && isreal(fitness)==1
            personal_best(g, :) = particles_position(g, :);
            personal_best_fitness(g) = fitness;
        end
        
        % 更新全局�?佳位�?
        if fitness < global_best_fitness && fitness>0 && isreal(fitness)==1
            global_best = particles_position(g, :);
            global_best_fitness = fitness;
       
        end
    end

    % Record data at each iteration
    all_particle_positions(iteration+1, :, :) = particles_position;
    all_global_best_positions(iteration+1, :) = global_best;
    all_global_best_fitness(iteration+1) = global_best_fitness;
    
    % 判断是否连续代数中最优个体未改变
    if iteration > 1 && isequal(global_best, previous_global_best)
        consecutive_stagnation = consecutive_stagnation + 1;
    else
        consecutive_stagnation = 0;
    end
    
    % 更新前一代的全局�?佳位�?
    previous_global_best = global_best;
    
    % 如果连续代数中最优个体未改变超过阈�?�，终止算法
    if consecutive_stagnation >= stagnation_threshold
        [fitness,final_Kernel_selection] = compute_fitness(global_best, num_views, numclass,Kernel,Y,z);
        
        break;
    end
end


% 计算适应度函�?
function [fitness,KH_best] = compute_fitness(particle, num_views, numclass,Kernel,Y,z)
    % 解析粒子中的特征选择信息
    
    for i1 = 1:num_views
        KH(:,:,i1) = Kernel{i1,particle(i1)};
    end
    KH = kcenter(KH);
    KH = knorm(KH);
    [KHL] = LocalKernelCalculation(KH  , 0.05);
    LargestIteration = 15;
    KernelWeight = ones(1 , num_views) / num_views;
    avgKer=sumKbeta(KHL,KernelWeight);
    BaseValue = norm(avgKer , 'fro')^2;
    RegularizationValue = 10^(-4) * BaseValue;
    Alpha_T = 0 * BaseValue;
    %Beta_T = 2.^[-8 , -2 , 2 , 6];
    Beta_T = 2.^[ -2 , 2 , 4];
    %NNRate = [0.01 , 0.03 , 0.09 , 0.11];
    NNRate = [0.01 , 0.03 , 0.11];
    lanbad =0.1;
    SampleNum = size(KH,1);
    LowRankRate = 0.05;
    acc=1;
    for IRate = 1 : length(NNRate)
        CRate = NNRate(IRate);
        [KHL , M1] = LGMKCPreprocessFinal(KH , CRate);

         for IBeta = 1 : length(Beta_T)
              Beta_C = Beta_T(IBeta);
              InputKnum = max(2* numclass , round(LowRankRate * SampleNum) );
              %[Mu , Z , flag , TotalObj] = LGMKCNew1(KHL, M1, RegularizationValue ,lanbad, zeros(SampleNum) , LargestIteration , InputKnum);
              [Mu , Z , flag , TotalObj] = LGMKCNew(KHL, M1, RegularizationValue , Alpha_T , Beta_C , zeros(SampleNum) , LargestIteration , InputKnum);
              PON = TotalObj(1:end-1) - TotalObj(2 : end);
              DON = sum(PON < -10^(-10)) == 0;
              PI = Z > 0;
              Z = Z.*PI;
              [U7] = baseline_spectral_onkernel( abs( (Z + Z') / 2) , numclass);
              U_normalized=U7./ repmat(sqrt(sum(U7.^2, 2)), 1,numclass);
              indx = litekmeans(U_normalized,numclass, 'MaxIter',100, 'Start',z,'Replicates',30);
              group = num2str(indx);
              group1 = num2cell(group);
              [p] = MatSurv(Y(:,1),Y(:,2),group1,'CensorLineLength',0,'LineWidth',1.2,'CensorLineWidth',1,'NoPlot','true');
              if p<acc
                  acc=p;
                  KH_best =KH;
              end
              
         end
  
   end

    % 根据选择的核函数计算适应度，可以使用聚类准确度作为�?�应�?
    fitness = acc;
end

% 随机生成每个视图中�?�择的特征数量，确保其合为m
function feature_selection = random_Kernel_selection(num_views,max_features)
    feature_selection = zeros(1, num_views);
    for v = 1:num_views
        feature_selection(v) = randi([1, max_features]);
  
    end
end

time = toc; % stop timer and calculate elapsed time

end

