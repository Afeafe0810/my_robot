close all
% 讀取 CSV 文件
data = readmatrix('alip_traj_output.csv');

% 提取時間和數據
t = data(:, 1);       % 時間
com_x = data(:, 2);   % com 的 x
com_y = data(:, 3);   % com 的 y
com_z = data(:, 4);   % com 的 z
lf_x = data(:, 5);    % lf 的 x
lf_y = data(:, 6);    % lf 的 y
lf_z = data(:, 7);    % lf 的 z
rf_x = data(:, 8);    % rf 的 x
rf_y = data(:, 9);    % rf 的 y
rf_z = data(:, 10);   % rf 的 z

% 計算軸範圍
x_min = min([com_x; lf_x; rf_x]) - 0.1;
x_max = max([com_x; lf_x; rf_x]) + 0.1;
y_min = min([com_y; lf_y; rf_y]) - 0.1;
% y_min = -1
y_max = max([com_y; lf_y; rf_y]) + 0.1;
% y_max = 1
% z_min = min([com_z; lf_z; rf_z]) - 0.1;
z_min = 0
z_max = max([com_z; lf_z; rf_z]) + 0.1;

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
    plot3(com_x(i), com_y(i), com_z(i), 'bo', 'MarkerSize', 8, 'LineWidth', 1.5); % COM
    plot3(lf_x(i), lf_y(i), lf_z(i), 'go', 'MarkerSize', 8, 'LineWidth', 1.5);  % LF
    plot3(rf_x(i), rf_y(i), rf_z(i), 'ro', 'MarkerSize', 8, 'LineWidth', 1.5);  % RF
    
    % 繪製歷史軌跡
    plot3(com_x(1:i), com_y(1:i), com_z(1:i), 'b-', 'LineWidth', 1.5); % COM
    plot3(lf_x(1:i), lf_y(1:i), lf_z(1:i), 'g-', 'LineWidth', 1.5);  % LF
    plot3(rf_x(1:i), rf_y(1:i), rf_z(1:i), 'r-', 'LineWidth', 1.5);  % RF
    legend({'COM', 'LF', 'RF'}, 'Location', 'best'); % 設置 Legend 標籤
    % 暫停以模擬動態效果
    pause(0.01);
    % input('go')
end

hold off;
