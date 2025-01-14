online_planning_dynamic()
%%
function online_planning_dynamic()
    % 機器人參數
    m = 9; % 機器人質量
    H = 0.45; % 理想質心高度
    W = 0.2; % 兩腳底間距
    g = 9.81; % 重力加速度
    l = sqrt(g / H); % 自然頻率
    % 理想參數
    T = 0.5; % 支撐間隔時長


    % 初始條件設置
    stance = 1; % 當前支撐腳：1為左腳，0為右腳
    contact_t = 0:0.01:0.5; % 支撐狀態運行時間
    P_cf_wf = [0; 0.1; 0]; % 支撐腳在地面座標系中的位置
    X0 = [0.0; 0]; % x方向初始狀態 (xc(0), ly(0))
    Y0 = [-0.1; -(0.5 * m * H * W) * (l * sinh(l * T)) / (1 + cosh(l * T))]; % y方向初始狀態 (yc(0), lx(0))
    P_Psw2com_0 = [0.0; 0.1]; % 擺動腳初始相對於質心的位置

    

    
    Vx_des_2T = 0; % 下一步速度
    Ly_des_2T = m * Vx_des_2T * H; % y方向理想角動量
    Lx_des_2T_1 = (0.5 * m * H * W) * (l * sinh(l * T)) / (1 + cosh(l * T));
    Lx_des_2T_2 = -(0.5 * m * H * W) * (l * sinh(l * T)) / (1 + cosh(l * T));
    zCL = 0.1; % 擺動腳高度參數（加大以強調Z方向）

    % 初始化 3D 圖
    figure;
    hold on;
    grid on;
    axis equal;
    xlabel('X (m)');
    ylabel('Y (m)');
    zlabel('Z (m)');
    title('ALIP Model: Dynamic 3D Trajectories');
    view(3); % 3D 視角

    % 初始化圖形元素
    CoM_plot = plot3(0, 0, 0, 'r-', 'LineWidth', 2); % 質心軌跡
    Swing_plot = plot3(0, 0, 0, 'b--', 'LineWidth', 1.5); % 擺動腳軌跡
    Support_plot = plot3(0, 0, 0, 'go', 'MarkerSize', 10); % 支撐腳位置

    % 主迴圈：逐步更新動態軌跡
    Com_trajectory = [];
    Swing_trajectory = [];
    Support_trajectory = [];

    for t = contact_t
        % ALIP 模型動態
        input('go')
        ALIP_x = [cosh(l * t), sinh(l * t) / (m * H * l);
                  m * H * l * sinh(l * t), cosh(l * t)];
        ALIP_y = [cosh(l * t), -sinh(l * t) / (m * H * l);
                  -m * H * l * sinh(l * t), cosh(l * t)];

        % 質心參考軌跡 (ALIP 求解)
        Xx_cf = ALIP_x * X0;
        Xy_cf = ALIP_y * Y0;
        Com_x_cf = Xx_cf(1);
        Com_y_cf = Xy_cf(1);
        Com_z_cf = H; % 質心高度波動

        % 質心參考軌跡在世界座標系下的位置
        Com_ref_wf = [Com_x_cf; Com_y_cf; Com_z_cf] + P_cf_wf;
        Com_trajectory = [Com_trajectory, Com_ref_wf];

        % 更新角動量
        Ly_T = m * H * l * sinh(l * T) * X0(1) + cosh(l * T) * X0(2);
        Lx_T = -m * H * l * sinh(l * T) * Y0(1) + cosh(l * T) * Y0(2);
        if stance == 1
            Lx_des_2T = Lx_des_2T_2;
        else
            Lx_des_2T = Lx_des_2T_1;
        end

        % 擺動腳軌跡
        pv = t / T; % 插值參數
        Psw2com_x_T = (Ly_des_2T - cosh(l * T) * Ly_T) / (m * H * l * sinh(l * T));
        Psw2com_y_T = (Lx_des_2T - cosh(l * T) * Lx_T) / (-m * H * l * sinh(l * T));
        [0.5 * ((1 + cos(pi * pv)) * P_Psw2com_0(1) - (1 - cos(pi * pv)) * Psw2com_x_T);
            0.5 * ((1 + cos(pi * pv)) * P_Psw2com_0(2) - (1 - cos(pi * pv)) * Psw2com_y_T);]
        Sw_x_cf = Com_x_cf - 0.5 * ((1 + cos(pi * pv)) * P_Psw2com_0(1) + (1 - cos(pi * pv)) * Psw2com_x_T);
        Sw_y_cf = Com_y_cf - 0.5 * ((1 + cos(pi * pv)) * P_Psw2com_0(2) + (1 - cos(pi * pv)) * Psw2com_y_T);
        Sw_z_cf = Com_z_cf - (4 * zCL * (pv - 0.5)^2 + (H - zCL));
        Swing_ref_wf = [Sw_x_cf; Sw_y_cf; Sw_z_cf] + P_cf_wf;
        Swing_trajectory = [Swing_trajectory, Swing_ref_wf];

        % 支撐腳軌跡
        Support_trajectory = [Support_trajectory, P_cf_wf];

        % 動態更新圖形
        set(CoM_plot, 'XData', Com_trajectory(1, :), 'YData', Com_trajectory(2, :), 'ZData', Com_trajectory(3, :));
        set(Swing_plot, 'XData', Swing_trajectory(1, :), 'YData', Swing_trajectory(2, :), 'ZData', Swing_trajectory(3, :));
        set(Support_plot, 'XData', Support_trajectory(1, :), 'YData', Support_trajectory(2, :), 'ZData', Support_trajectory(3, :));

        pause(0.01); % 暫停以顯示動畫效果
    end

    hold off;
end