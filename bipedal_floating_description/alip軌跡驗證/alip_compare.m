clear; clc; close all

Norman_dir = "/home/ldsc/ros2_ws/src/bipedal_floating_description/bipedal_floating_description/alip軌跡驗證/祐華學長ALIP/";
% online的理論值
my_ref = readtable('real_planning.csv');

% 實際的量測值
my_mea = readtable('real_measure.csv');

% 祐華學長的理論值
Norman_ref = readtable(Norman_dir +'ref.csv');

%==========第一張圖==========%
labels = { ...
    'com_y', 'com_z', ...
    'lf_y', 'lf_z', ...
    'rf_y', 'rf_z', ...
    'y', 'Lx', ...
};
figure;
    for i = 1:length(labels)
        subplot(4, 2, i);
        hold on;
    
        label = labels{i};
        plot(my_ref.(label), "o");
        plot(Norman_ref.(label), ".");
        plot(my_mea.(label), ".");
        
        title(label);
        legend("my ref", "norman ref", "measure");
        hold off;
    end
%==========第二張圖==========%
labels = {"pel_y","pel_z"};
figure;
    for i = 1:length(labels)
        subplot(3, 1, i);
        hold on;
    
        label = labels{i};
        plot(my_ref.(label), "o");
        plot(my_mea.(label), ".");
        
        title(label);
        legend("my ref", "measure");
        hold off;
    end
