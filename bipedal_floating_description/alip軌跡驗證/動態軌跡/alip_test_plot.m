close all; clear; clc
% 讀取 CSV 文件
data = readmatrix('real_planning.csv');
data= data(2:end,2:end)
data1 = readmatrix('real_measure.csv');
data1= data1(2:end,2:end)
% 提取時間和數據
pel_x = data(:, 1);   % com 的 x
pel_y = data(:, 2);   % com 的 y
pel_z = data(:, 3);   % com 的 z
lf_x = data(:, 4);    % lf 的 x
lf_y = data(:, 5);    % lf 的 y
lf_z = data(:, 6);    % lf 的 z
rf_x = data(:, 7);    % rf 的 x
rf_y = data(:, 8);    % rf 的 y
rf_z = data(:, 9);   % rf 的 z
x = data(:, 10);
y = data(:, 11);
Ly = data(:, 12);
Lx = data(:, 13);

pel_x_m = data1(:, 1);   % com 的 x
pel_y_m = data1(:, 2);   % com 的 y
pel_z_m = data1(:, 3);   % com 的 z
lf_x_m = data1(:, 4);    % lf 的 x
lf_y_m = data1(:, 5);    % lf 的 y
lf_z_m = data1(:, 6);    % lf 的 z
rf_x_m = data1(:, 7);    % rf 的 x
rf_y_m = data1(:, 8);    % rf 的 y
rf_z_m = data1(:, 9);   % rf 的 z
x_m = data1(:, 10);
y_m = data1(:, 11);
Ly_m = data1(:, 12);
Lx_m = data1(:, 13);

t = ( 0:length(pel_x)-1 ) *0.01;
figure
    hold on
    plot(pel_z,'.')
    plot(pel_z_m,'o')
    legend('ref', 'mea')

%%
% 計算軸範圍
x_min = min([pel_x; lf_x; rf_x]) - 0.1;
x_max = max([pel_x; lf_x; rf_x]) + 0.1;
y_min = min([pel_y; lf_y; rf_y]) - 0.1;
% y_min = -1
y_max = max([pel_y; lf_y; rf_y]) + 0.1;
% y_max = 1
% z_min = min([com_z; lf_z; rf_z]) - 0.1;
z_min = 0
z_max = max([pel_z; lf_z; rf_z]) + 0.1;

% 初始化 3D 圖
figure;
hold on;
grid on;
xlabel('X-axis (m)');
ylabel('Y-axis (m)');
zlabel('Z-axis (m)');
title('3D Dynamic Curve of COM, LF, and RF');
view(3);

% 設定固定軸範圍
axis([x_min, x_max, y_min, y_max, z_min, z_max]);

% 動態繪圖
com_line = plot3(NaN, NaN, NaN, 'b-', 'LineWidth', 1.5); % Placeholder for COM
lf_line = plot3(NaN, NaN, NaN, 'g-', 'LineWidth', 1.5); % Placeholder for LF
rf_line = plot3(NaN, NaN, NaN, 'r-', 'LineWidth', 1.5); % Placeholder for RF


for i = 1:length(t)
    % 清空之前的圖層
    cla;
    
    % 繪製當前位置
    plot3(pel_x(i), pel_y(i), pel_z(i), 'bo', 'MarkerSize', 8, 'LineWidth', 1.5); % COM
    plot3(lf_x(i), lf_y(i), lf_z(i), 'go', 'MarkerSize', 8, 'LineWidth', 1.5);  % LF
    plot3(rf_x(i), rf_y(i), rf_z(i), 'ro', 'MarkerSize', 8, 'LineWidth', 1.5);  % RF
    
    % 繪製歷史軌跡
    plot3(pel_x(1:i), pel_y(1:i), pel_z(1:i), 'b-', 'LineWidth', 1.5); % COM
    plot3(lf_x(1:i), lf_y(1:i), lf_z(1:i), 'g-', 'LineWidth', 1.5);  % LF
    plot3(rf_x(1:i), rf_y(1:i), rf_z(1:i), 'r-', 'LineWidth', 1.5);  % RF
    legend({'COM', 'LF', 'RF'}, 'Location', 'best'); % 設置 Legend 標籤
    % 暫停以模擬動態效果
    pause(0.01);
    % if round(t(i),3) == 0.55
    %     input('wait...')
    % end
end

hold off;
