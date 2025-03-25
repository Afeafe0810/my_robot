%%
COM = [0;0;0.57];
LF = [0;0.1;0];
RF = [0;-0.1;0];

figure;
scatter3(COM(1,1),COM(2,1),COM(3,1),"filled"); 
hold on;
scatter3(LF(1,1),LF(2,1),LF(3,1),"filled"); 
scatter3(RF(1,1),RF(2,1),RF(3,1),"filled"); 

line([COM(1,1); LF(1,1)], [COM(2,1); LF(2,1)], [COM(3,1); LF(3,1)], 'Color', 'black');
line([COM(1,1); RF(1,1)], [COM(2,1); RF(2,1)], [COM(3,1); RF(3,1)], 'Color', 'black');

%%
%Initial state
X0 = [0.0;0.]; %xc ly
psw2com_x = 0.0;
psw2com_x_past = 0.0;

Y0 = [-0.025;0.0]; %yc lx
psw2com_y = 0.0;
psw2com_y_past = 0.18;  %need to modify

time = 0;
t0 = 0; 

%Robot state
m = 9;
H = 0.45; %ALIP_height
pelvis_H = 0.55
g = 9.81;
l = sqrt(g/H);

%Time
T = 0.5; %Step time
time_step = 0.025; %Time Step
max_time = 2.5; %Max Time

%Desired AM
vdes = 0.;
Lydes = m*vdes*H;

LX_DES_1 = 0.5*m*H*0.2*l*sinh(l*T)/(1+cosh(l*T));
LX_DES_2 = -0.5*m*H*0.2*l*sinh(l*T)/(1+cosh(l*T));
LX_DES = [LX_DES_1;LX_DES_2];

scatter3(RF(1,1),RF(2,1),RF(3,1),"filled"); 
%for plot data
LF_x_wf = 0;
RF_x_wf = 0;
COM_x_wf = X0(1,1) + LF_x_wf; %initial support by left leg

LF_y_wf = 0.1;
RF_y_wf = -0.1;
COM_y_wf = Y0(1,1) + LF_y_wf; %initial support by left leg

contact = 0;

%for trajectory
zCL = 0.02;
figure;

%for trajectory data collect (all in wf)
C_contact = [];
alip_t = [];
cx = []; %質心在大地座標的位置
cy = [];
cz = [];
l_leg_x = [];
l_leg_y = [];
l_leg_z = [];
r_leg_x = [];
r_leg_y = [];
r_leg_z = [];
xc = [];%質心在cf的位置
yc = [];
ly = [];
lx = [];



for t = 0:time_step:max_time

    %ALIP_in_x_direction
    ALIP_x = [cosh(l*(t-t0)) sinh(l*(t-t0))/(m*H*l);
        m*H*l*sinh(l*(t-t0)) cosh(l*(t-t0))];
    %ALIP_in_y_direction
    ALIP_y = [cosh(l*(t-t0)) -sinh(l*(t-t0))/(m*H*l);
        -m*H*l*sinh(l*(t-t0)) cosh(l*(t-t0))];

    %calculate xc ly
    Xx = ALIP_x * X0;
    %calculate yc lx
    Xy = ALIP_y * Y0;

    %update the angular momentum at the end of current step
    ly_hat = m*H*l*sinh(l*(T))*X0(1,1) + cosh(l*(T))*X0(2,1);
    %update the angular momentum at the end of current step
    lx_hat = -m*H*l*sinh(l*(T))*Y0(1,1) + cosh(l*(T))*Y0(2,1);

    %change Ldes
    if mod(contact,2) == 0
        Lxdes = LX_DES(2,1);
    else
        Lxdes = LX_DES(1,1);
    end

    %calculate desired swing foot position of next step
    psw2com_x = (Lydes - cosh(l*T)*ly_hat) / (m*H*l*sinh(l*T));
    %calculate desired swing foot position of next step
    psw2com_y = (Lxdes - cosh(l*T)*lx_hat) / -(m*H*l*sinh(l*T));

    % calculate COM_x LF_x RF_x in world frame
    [COM_x_wf,LF_x_wf,RF_x_wf,psw2com_x_past] = X_wf(LF_x_wf,RF_x_wf,Xx,psw2com_x_past,psw2com_x,contact,t,t0,T);
    % % calculate COM_y LF_y RF_y in world frame
    [COM_y_wf,LF_y_wf,RF_y_wf,psw2com_y_past] = Y_wf(LF_y_wf,RF_y_wf,Xy,psw2com_y_past,psw2com_y,contact,t,t0,T);
    % calculate COM_z LF_z RF_z in world frame
    [COM_z_wf,LF_z_wf,RF_z_wf] = Z_wf(contact,t,t0,T,pelvis_H,zCL);

    % scatter(t,Xy(2,1));
    % scatter(t,COM_y_wf);
    scatter3(COM_x_wf,COM_y_wf,COM_z_wf,'filled','o','black');
    scatter3(LF_x_wf,LF_y_wf,LF_z_wf,'filled','o','red');
    scatter3(RF_x_wf,RF_y_wf,RF_z_wf,'filled','o','blue');
    CL = line([COM_x_wf; LF_x_wf], [COM_y_wf; LF_y_wf], [COM_z_wf; LF_z_wf], 'Color', 'red');
    CR = line([COM_x_wf; RF_x_wf], [COM_y_wf; RF_y_wf], [COM_z_wf; RF_z_wf], 'Color', 'blue');
    pause(time_step);
    if mod(t,T) ~= 0
        delete(CL);
        delete(CR);
    else
        fprintf('t= \n',t)
        Xy(2,1)
    end
    xlim([-0.2 0.6]);
    ylim([-0.2 0.2]);
    zlim([0 0.8]);
    hold on;

    %dtat collect
    C_contact = [C_contact;contact];
    alip_t = [alip_t;t];
    cx = [cx;COM_x_wf];
    cy = [cy;COM_y_wf];
    cz = [cz;COM_z_wf];
    l_leg_x = [l_leg_x;LF_x_wf];
    l_leg_y = [l_leg_y;LF_y_wf];
    l_leg_z = [l_leg_z;LF_z_wf];
    r_leg_x = [r_leg_x;RF_x_wf];
    r_leg_y = [r_leg_y;RF_y_wf];
    r_leg_z = [r_leg_z;RF_z_wf];
    xc = [xc;Xx(1,1)];
    yc = [yc;Xy(1,1)];
    ly = [ly;Xx(2,1)];
    lx = [lx;Xy(2,1)];
   
    %update the frame & contact & t0
    if mod(t,T) == 0 & t>0
        %update xc ly yc lx t0
        X0 = [psw2com_x;Xx(2,1)];
        Y0 = [psw2com_y;Xy(2,1)];
        t0 = t;
        contact = contact + 1;
    end
end

%function for plot COM_x LF_x RF_x in world frame
function [COM_x_wf,LF_x_wf,RF_x_wf,psw2com_x_past] = X_wf(LF_x_wf,RF_x_wf,Xx,psw2com_x_past,psw2com_x,contact,t,t0,T)

    pv = (t-t0)/T;

    if mod(contact,2) == 0 %left support
        COM_x_wf = LF_x_wf + Xx(1,1);
        LF_x_wf = LF_x_wf;
        RF_x_wf = COM_x_wf - (0.5*((1+cos(pi*pv))*psw2com_x_past + (1-cos(pi*pv))*psw2com_x));

        if mod(t,T) == 0 && t>0
            RF_x_wf = COM_x_wf - psw2com_x;
            psw2com_x_past = Xx(1,1);
        end

    else %right support
        COM_x_wf = RF_x_wf + Xx(1,1);
        LF_x_wf = COM_x_wf - (0.5*((1+cos(pi*pv))*psw2com_x_past + (1-cos(pi*pv))*psw2com_x));
        RF_x_wf = RF_x_wf;

        if mod(t,T) == 0 & t>0
            LF_x_wf = COM_x_wf - psw2com_x;
            psw2com_x_past = Xx(1,1);
        end
    end
end

%function for plot COM_x LF_x RF_x in world frame
function [COM_y_wf,LF_y_wf,RF_y_wf,psw2com_y_past] = Y_wf(LF_y_wf,RF_y_wf,Xy,psw2com_y_past,psw2com_y,contact,t,t0,T)

    pv = (t-t0)/T;

    if mod(contact,2) == 0 %left support
        COM_y_wf = LF_y_wf + Xy(1,1);
        LF_y_wf = LF_y_wf;
        RF_y_wf = COM_y_wf - (0.5*((1+cos(pi*pv))*psw2com_y_past + (1-cos(pi*pv))*psw2com_y));

        if mod(t,T) == 0 && t>0
            RF_y_wf = COM_y_wf - psw2com_y;
            psw2com_y_past = Xy(1,1);
        end

    else %right support
        COM_y_wf = RF_y_wf + Xy(1,1);
        LF_y_wf = COM_y_wf - (0.5*((1+cos(pi*pv))*psw2com_y_past + (1-cos(pi*pv))*psw2com_y));
        RF_y_wf = RF_y_wf;

        if mod(t,T) == 0 & t>0
            LF_y_wf = COM_y_wf - psw2com_y;
            psw2com_y_past = Xy(1,1);
        end
    end
end

%function for plot COM_z LF_z RF_z in world frame
function [COM_z_wf,LF_z_wf,RF_z_wf] = Z_wf(contact,t,t0,T,H,zCL)

    pv = (t-t0)/T;

    if mod(contact,2) == 0 %left support
        COM_z_wf = H;
        LF_z_wf = 0;
        RF_z_wf = H - (4*zCL*(pv-0.5)^2 + (H-zCL));

    else %right support
        COM_z_wf = H;
        LF_z_wf = H - (4*zCL*(pv-0.5)^2 + (H-zCL));
        RF_z_wf = 0;

    end
end



