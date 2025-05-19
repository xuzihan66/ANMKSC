function [final_Kernel_selection, all_particle_positions, all_global_best_positions, all_global_best_fitness, iteration,time] = AOW1(Kernel,numclass, Y, z,num_views,m, w, c1,c2)

%% Include dependencies
addpath('./lib'); % dependencies
addpath('./methods'); % FS methods
addpath(genpath('./lib/drtoolbox'));

tic; % start timer


% 参数设置

num_particles =40; % 粒子数量
max_iterations = 40; % 最大迭代次数
v= num_views ; % 视图数量
% num_features_per_view = m; % 每个视图已选择的已排序特征数量，这里假设每个视图选取m个特征
% w = 0.9; % 惯性权重
% c1 = 2; % 个体学习因子
% c2 = 2; % 社会学习因子
% 初始化粒子群
particles_position = zeros(num_particles, num_views); % 初始化粒子位置，每个视图选择的特征数量
particles_velocity = rand(num_particles, num_views); % 初始化粒子速度
personal_best = particles_position; % 个体最佳位置初始化为当前位置
global_best = []; % 全局最佳位置
consecutive_stagnation = 0; % 连续代数中最优个体未改变的计数器
stagnation_threshold = 5; % 设定连续代数未改变的阈值

% 跟踪最佳个体和全局的适应度
personal_best_fitness = zeros(1, num_particles);
global_best_fitness = -inf; % 初始全局最佳适应度设为负无穷

% Initialize variables to store the data
all_particle_positions = zeros(max_iterations+1, num_particles, num_views);
all_global_best_positions = zeros(max_iterations+1, num_views);

%------------------------------------------------------------------------------
%聚类参数设置
    options.seuildiffsigma=1e-5;        % stopping criterion for weight variation
%------------------------------------------------------
% Setting some numerical parameters
%------------------------------------------------------
    options.goldensearch_deltmax=1e-3; % initial precision of golden section search
    options.numericalprecision=1e-16;   % numerical precision weights below this value
% are set to zero
%------------------------------------------------------
% some algorithms paramaters
%------------------------------------------------------
    options.firstbasevariable='first'; % tie breaking method for choosing the base
% variable in the reduced gradient method
    options.nbitermax=500;             % maximal number of iteration
    options.seuil=0;                   % forcing to zero weights lower than this
    options.seuilitermax=10;           % value, for iterations lower than this one
    options.miniter=0;                 % minimal number of iterations
    options.threshold = 1e-4;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% 计算初始适应度并更新全局最佳位置
for g = 1:num_particles
    % 随机初始化粒子位置，确保每个视图中选择的特征数量合为m
    particles_position(g, :) = random_Kernel_selection(num_views, 20);
    
    % 计算适应度，包括分类准确度
    [fitness] = compute_fitness(particles_position(g, :), num_views,numclass,Kernel,Y,z);

    % 记录个体最佳适应度
    personal_best_fitness(g) = fitness;
    
    % 更新个体最佳位置
    personal_best(g, :) = particles_position(g, :);
    
    % 更新全局最佳位置
    if  isempty(global_best) || (fitness < global_best_fitness && fitness>0 && isreal(fitness)==1)
        global_best = particles_position(g, :);
        global_best_fitness = fitness;
  
    end
end

all_particle_positions(1, :, :) = particles_position;
all_global_best_positions(1, :) = global_best;
all_global_best_fitness(1) = global_best_fitness;

previous_global_best = global_best; % 初始化前一代的全局最佳位置

% 开始迭代
for iteration = 1:max_iterations
    for g = 1:num_particles
        % 计算速度更新
        particles_velocity(g, :) = w * particles_velocity(g, :) ...
            + c1 * rand() * (personal_best(g, :) - particles_position(g, :)) ...
            + c2 * rand() * (global_best - particles_position(g, :));
        
        % 限制速度范围，确保在合理的范围内
        particles_velocity(g, :) = min(max(particles_velocity(g, :), -1), 1);
        
        % 更新粒子位置（使用 round 函数来确保整数值）
        particles_position(g, :) = round(particles_position(g, :) + particles_velocity(g, :));
        particles_position(g, :) = max(particles_position(g, :), 0);
        for i=1:3
            if particles_position(g,i)== 0
                particles_position(g,i)=particles_position(g,i)+1;
            elseif particles_position(g,i)== 21
                    particles_position(g,i)=particles_position(g,i)-1;
            end
            
        end
%        % 随机修正粒子位置，以确保每个视图中选择的特征数量合为m
%        particles_position(i, :) = random_feature_correction(particles_position(i, :), num_views, m);
        
        % 计算适应度，包括分类准确度
        [fitness] = compute_fitness(particles_position(g, :), num_views,numclass, Kernel,Y,z);
        
        % 更新个体最佳位置
        if fitness < personal_best_fitness(g) && fitness>0  && isreal(fitness)==1
            personal_best(g, :) = particles_position(g, :);
            personal_best_fitness(g) = fitness;
        end
        
        % 更新全局最佳位置
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
    
    % 更新前一代的全局最佳位置
    previous_global_best = global_best;
    
    % 如果连续代数中最优个体未改变超过阈值，终止算法
    if consecutive_stagnation >= stagnation_threshold
        [fitness,final_Kernel_selection] = compute_fitness(global_best, num_views, numclass,Kernel,Y,z);
        
        break;
    end
end


% 计算适应度函数
function [fitness,KH_best] = compute_fitness(particle, num_views, numclass,Kernel,Y,z)
    % 解析粒子中的特征选择信息
    
    for i1 = 1:num_views
        KH(:,:,i1) = Kernel{i1,particle(i1)};
    end

    numker = size(KH,3);
    num = size(KH,1);
    KH = kcenter(KH);
    KH = knorm(KH);

    qnorm = 2;
    
    tic;
    [U7,Sigma,obj] = simpleMKKM(KH,numclass,options);
    U_normalized=U7./ repmat(sqrt(sum(U7.^2, 2)), 1,numclass);
    indx = litekmeans(U_normalized,numclass, 'MaxIter',100, 'Start',z,'Replicates',30);
    group = num2str(indx);
    group1 = num2cell(group);
    [p] = MatSurv(Y(:,1),Y(:,2),group1,'CensorLineLength',0,'LineWidth',1.2,'CensorLineWidth',1,'NoPlot','true');
     fitness=p;
     KH_best =KH;
    
              

end

% 随机生成每个视图中选择的特征数量，确保其合为m
function feature_selection = random_Kernel_selection(num_views,max_features)
    feature_selection = zeros(1, num_views);
    for v = 1:num_views
        feature_selection(v) = randi([1, max_features]);
  
    end
end

% % 随机修正粒子位置，以确保每个视图中选择的特征数量合为m
% function feature_selection = random_feature_correction(particle, num_views, m)
%     current_selection = particle; % 当前的特征选择
%     excess = sum(current_selection) - m;
%     if excess > 0
%         num_selected = sum(current_selection);
%         while num_selected > m
%             % 随机选择已选择的特征，直到特征数量合为m
%             selected_features = find(current_selection);
%             selected = randsample(selected_features, 1);
%             current_selection(selected) = current_selection(selected)-1;
%             num_selected = num_selected - 1;
%         end
%     elseif excess < 0
%         num_selected = sum(current_selection);
%         while num_selected < m
%             % 随机选择已选择的特征，直到特征数量合为m
%             selected = randsample(num_views, 1);
%             current_selection(selected) = current_selection(selected)+1;
%             num_selected = num_selected + 1;
%         end
%     end
%     feature_selection = current_selection;
% end

time = toc; % stop timer and calculate elapsed time

end

