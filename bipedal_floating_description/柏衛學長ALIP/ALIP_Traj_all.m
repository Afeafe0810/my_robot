%% https://github.com/UMich-BipedLab/Cassie_Controller_AngularMomentum/blob/main/CustomCodes/Controllers_Estimators/%40Cassie_Controller_4/Cassie_Controller_4.m#L694
clear;close;

%%
R=@(t)([cosd(t) -sind(t) ; ...
        sind(t) cosd(t)]);
m = 9;
g = 9.81;
H = 0.45;
T = 0.5; 
l = sqrt(g/H);
C = cosh(l*T);
S = sinh(l*T);
ts = 0.01;
zCL = 0.02;

W = 0.2;
v = 0.3;
%%
Lyt2=[	0 0 0 0 0 0 ];																															

theta=zeros(1,length(Lyt2)+2				);
theta(4)=0.0;
% for i=1:length(theta)/2
%     theta(i)=5*i-5;
% end
% for i=length(theta)/2:length(theta)
%     theta(i)=60-5*i;
% end
% theta(13)=theta(12);
% theta(13)=theta(12);
% theta=rad2deg(theta);
swd = theta(2);
px_rf=0;
py_rf=0.1;
goalx=0.5;
goaly=1;
tgd=0;
%%
x=0;
y=-0.00;
yaw=0;
px=0.0;
py=-0.1;
%----------------------------%
% theta=rad2deg(theta);
swd = theta(2);
Lyt2=Lyt2*m*H*v;
%%
xt=x(1);
yt=y(1);
yaw = theta(1);

pxt_yaw = 0;
pyt_yaw = W/2;

[pxt,pyt]=rot(pxt_yaw,pyt_yaw,yaw);

pxf = px(1);
pyf = py(1);

pxt=xt-pxt;
pyt=yt-pyt;

Lyt_yaw = 0.00;
Lxt_yaw = 0.5*(m*H*l*S)*W/(1+C);
% Lxt_yaw=0;
xt_collect=[xt];
yt_collect=[yt];

px_collect=[pxt];
py_collect=[pyt];
theta_collect=[theta(1)];

xc_collect=[];
yc_collect=[];
ly_collect=[];
lx_collect=[];

lf_x_collect=[];
lf_y_collect=[];
lf_z_collect=[];
rf_x_collect=[];
rf_y_collect=[];
rf_z_collect=[];

is_left_support=0;

%%
Lyt2_swd = Lyt2(1);
Lxt2_swd = 0.5*m*H*W*(l*S)/(1+C);

Lyt1_yaw = (m*H*l*S)*pxt_yaw+C*Lyt_yaw;
Lxt1_yaw = -(m*H*l*S)*pyt_yaw+C*Lxt_yaw;

[Lyt1_swd,Lxt1_swd]=rot(Lyt1_yaw,Lxt1_yaw,yaw-swd);

pxt1_swd=(Lyt2_swd-C*Lyt1_swd)/(m*H*l*S);
pyt1_swd=(Lxt2_swd-C*Lxt1_swd)/-(m*H*l*S);

[pxt1,pyt1]=rot(pxt1_swd,pyt1_swd,swd);

psw2comx_past=0.1*sind(yaw);
psw2comx=pxt1;
psw2comy_past=-0.1*cosd(yaw);
psw2comy=pyt1;

x0=[pxt_yaw;Lyt_yaw];
y0=[pyt_yaw;Lxt_yaw];
t0=0;
for t = 0.00:0.01:0.5
    ALIP_x = [cosh(l*(t-t0))          sinh(l*(t-t0))/(m*H*l);...
              m*H*l*sinh(l*(t-t0))    cosh(l*(t-t0))];
    ALIP_y = [cosh(l*(t-t0))         -sinh(l*(t-t0))/(m*H*l);...
              -m*H*l*sinh(l*(t-t0))   cosh(l*(t-t0))];
    x_x=ALIP_x*x0;
    x_y=ALIP_y*y0;
    x_turn=R(yaw)*[x_x(1,1);x_y(1,1)];
    ly_collect=cat(2,ly_collect,x_x(2,1));
    lx_collect=cat(2,lx_collect,x_y(2,1));
    xc=x_turn(1,1)+pxf;
    yc=x_turn(2,1)+pyf;
    xc_collect=cat(2,xc_collect,xc);
    yc_collect=cat(2,yc_collect,yc);
    
    pv = (t-t0)/T;
    % is_left_support==0
    lf_x=xc-(0.5*((1+cos(pi*pv))*psw2comx_past + (1-cos(pi*pv))*psw2comx));
    lf_y=yc-(0.5*((1+cos(pi*pv))*psw2comy_past + (1-cos(pi*pv))*psw2comy));
    lf_z=H-(4*zCL*(pv-0.5)^2 + (H-zCL));
    rf_x=px(1);
    rf_y=py(1);
    rf_z=0;

    lf_x_collect=cat(2,lf_x_collect,lf_x);
    lf_y_collect=cat(2,lf_y_collect,lf_y);
    lf_z_collect=cat(2,lf_z_collect,lf_z);
    rf_x_collect=cat(2,rf_x_collect,rf_x);
    rf_y_collect=cat(2,rf_y_collect,rf_y);
    rf_z_collect=cat(2,rf_z_collect,rf_z);
    
end


xt_yaw = C*pxt_yaw+(S/(m*H*l))*Lyt_yaw;
yt_yaw = C*pyt_yaw-(S/(m*H*l))*Lxt_yaw;
xt_turn = R(yaw)*[xt_yaw;yt_yaw];
xt=pxf+xt_turn(1);
yt=pyf+xt_turn(2);
pxf=xt-pxt1;
pyf=yt-pyt1;

is_left_support=1-is_left_support;

xt_collect=cat(2,xt_collect,xt);
yt_collect=cat(2,yt_collect,yt);
px_collect=cat(2,px_collect,pxf);
py_collect=cat(2,py_collect,pyf);
theta_collect=cat(2,theta_collect,swd);

%%
for i=1:length(Lyt2)-1
yaw=swd;
swd=theta(i+2);

Lyt_yaw=Lyt1_swd;
Lxt_yaw=Lxt1_swd;

pxt_yaw=pxt1_swd;
pyt_yaw=pyt1_swd;
[pxt,pyt]=rot(pxt_yaw,pyt_yaw,yaw);

pxf=xt-pxt;
pyf=yt-pyt;

Lyt2_swd=Lyt2(i+1);
Lxt2_swd=-Lxt2_swd;

Lyt1_yaw = (m*H*l*S)*pxt_yaw+C*Lyt_yaw;
Lxt1_yaw = -(m*H*l*S)*pyt_yaw+C*Lxt_yaw;
[Lyt1_swd,Lxt1_swd]=rot(Lyt1_yaw,Lxt1_yaw,yaw-swd);

pxt1_swd=(Lyt2_swd-C*Lyt1_swd)/(m*H*l*S);
pyt1_swd=(Lxt2_swd-C*Lxt1_swd)/-(m*H*l*S);

[pxt1,pyt1]=rot(pxt1_swd,pyt1_swd,swd);

psw2comx_past=x_turn(1,1);
psw2comx=pxt1;
psw2comy_past=x_turn(2,1);
psw2comy=pyt1;

x0=[pxt_yaw;Lyt_yaw];
y0=[pyt_yaw;Lxt_yaw];

t0=0;
for t = 0.0:0.01:0.5
    ALIP_x = [cosh(l*(t-t0))          sinh(l*(t-t0))/(m*H*l);...
              m*H*l*sinh(l*(t-t0))    cosh(l*(t-t0))];
    ALIP_y = [cosh(l*(t-t0))         -sinh(l*(t-t0))/(m*H*l);...
              -m*H*l*sinh(l*(t-t0))   cosh(l*(t-t0))];
    x_x=ALIP_x*x0;
    x_y=ALIP_y*y0;
    x_turn=R(yaw)*[x_x(1,1);x_y(1,1)];
    xc=x_turn(1,1)+pxf;
    yc=x_turn(2,1)+pyf;
    ly_collect=cat(2,ly_collect,x_x(2,1));
    lx_collect=cat(2,lx_collect,x_y(2,1));
    xc_collect=cat(2,xc_collect,xc);
    yc_collect=cat(2,yc_collect,yc);
    
    pv = (t-t0)/T;
    if is_left_support==0
        lf_x=xc-(0.5*((1+cos(pi*pv))*psw2comx_past + (1-cos(pi*pv))*psw2comx));
        lf_y=yc-(0.5*((1+cos(pi*pv))*psw2comy_past + (1-cos(pi*pv))*psw2comy));
        lf_z=H-(4*zCL*(pv-0.5)^2 + (H-zCL));
        rf_x=rf_x;
        rf_y=rf_y;
        rf_z=0;
    else
        rf_x=xc-(0.5*((1+cos(pi*pv))*psw2comx_past + (1-cos(pi*pv))*psw2comx));
        rf_y=yc-(0.5*((1+cos(pi*pv))*psw2comy_past + (1-cos(pi*pv))*psw2comy));
        rf_z=H-(4*zCL*(pv-0.5)^2 + (H-zCL));
        lf_x=lf_x;
        lf_y=lf_y;
        lf_z=0;
    end
    lf_x_collect=cat(2,lf_x_collect,lf_x);
    lf_y_collect=cat(2,lf_y_collect,lf_y);
    lf_z_collect=cat(2,lf_z_collect,lf_z);
    rf_x_collect=cat(2,rf_x_collect,rf_x);
    rf_y_collect=cat(2,rf_y_collect,rf_y);
    rf_z_collect=cat(2,rf_z_collect,rf_z);
end

xt_yaw=C*pxt_yaw+(S/(m*H*l))*Lyt_yaw;
yt_yaw=C*pyt_yaw-(S/(m*H*l))*Lxt_yaw;
xt_turn=R(yaw)*[xt_yaw;yt_yaw];
xt=pxf+xt_turn(1);
yt=pyf+xt_turn(2);

pxf=xt-pxt1;
pyf=yt-pyt1;

is_left_support=1-is_left_support;

xt_collect=cat(2,xt_collect,xt);
yt_collect=cat(2,yt_collect,yt);
px_collect=cat(2,px_collect,pxf);
py_collect=cat(2,py_collect,pyf);
theta_collect=cat(2,theta_collect,swd);
end
%%
figure();
title('Foot Placement and Trajectory')
xlabel('x (m)')
ylabel('y (m)')
zlabel('z (m)')
set(gcf,'Position',[100,100,1080,1080],'Color','w');
xlow=min(px_collect)-0.2;xhigh=max(px_collect)+0.2;ylow=min(py_collect)-0.2;yhigh=max(py_collect)+0.2;
axis equal;axis manual;axis([xlow xhigh ylow yhigh])
ax=gca;
ax.XGrid='on';
ax.YGrid='on';
hold on
view(-46,38) % use [caz,cel] = view
color = ['r','b'];
l_foot=0.16/2;

% start & goal
show_start_and_goal(xt_collect(1),yt_collect(1),theta(1),goalx,goaly,tgd,xhigh);

% init foot place
polypinit=polystep( px_rf, py_rf, theta(1) );
plot(polypinit,'FaceColor','w','EdgeColor',color(1),'LineStyle','-');
quiver(px_rf,py_rf,l_foot*cosd(theta(1)),l_foot*sind(theta(1)),AutoScaleFactor=1,MaxHeadSize=4,Marker=".",MarkerSize=4,Color=color(1))

i=1;
polyp(i) = polystep(px_collect(i), py_collect(i), theta_collect(i));
plot(polyp(i), 'FaceColor', 'w', 'EdgeColor', color(rem(i, 2) + 1), 'LineStyle', '-');
quiver(px_collect(i),py_collect(i),l_foot*cosd(theta(i)),l_foot*sind(theta(i)),AutoScaleFactor=1,MaxHeadSize=4,Marker=".",MarkerSize=4,Color=color(2))
pause(0.1)

j=2;
for i=1:length(lx_collect)
    plot3(xc_collect(i),yc_collect(i),H,'.-','Color','k',"LineWidth",0.5,'MarkerSize',10)
    plot3(lf_x_collect(i),lf_y_collect(i),lf_z_collect(i),'.-','Color','r',"LineWidth",0.5,'MarkerSize',10)
    plot3(rf_x_collect(i),rf_y_collect(i),rf_z_collect(i),'.-','Color','b',"LineWidth",0.5,'MarkerSize',10)
    CL = line([xc_collect(i); lf_x_collect(i)], [yc_collect(i); lf_y_collect(i)], [H;lf_z_collect(i)], 'Color', 'r');
    CR = line([xc_collect(i); rf_x_collect(i)], [yc_collect(i); rf_y_collect(i)], [H;rf_z_collect(i)], 'Color', 'b');

    if mod(i,T/ts+1)==0% && mod(i,T/ts/2)==0
        polyp(j) = polystep(px_collect(j), py_collect(j), theta_collect(j));
        plot(polyp(j), 'FaceColor', 'w', 'EdgeColor', color(rem(j, 2) + 1), 'LineStyle', '-');
        quiver(px_collect(j),py_collect(j),l_foot*cosd(theta(j)),l_foot*sind(theta(j)),AutoScaleFactor=1,MaxHeadSize=4,Marker=".",MarkerSize=4,Color=color(rem(j, 2)+1))
        j=j+1;
    end

    
    pause(0.00001)    
    if mod(i,T/ts+1) ~= 0
        delete(CL);
        delete(CR);
    end
end

writematrix([lx_collect;ly_collect]',"l_traj.csv")
writematrix([xc_collect;yc_collect;ones(1,length(xc_collect))*H]',"xc_traj.csv")
writematrix([rf_x_collect; rf_y_collect; rf_z_collect]', "rf_traj.csv")
writematrix([lf_x_collect; lf_y_collect; lf_z_collect]', "lf_traj.csv")
%%
function[a,b]= rot(x,y,theta)
%%rotation
    a=cosd(theta)*x-sind(theta)*y;
    b=sind(theta)*x+cosd(theta)*y;
end

function [outputpoly] = polystep(x,y,theta)
%%draw poly
w=0.09/2;
l=0.16/2;
polyin=polyshape([-l l l -l],[-w -w w w]);
outputpoly=rotate(translate(polyin,[x y]),theta,[x,y]);
end

function show_start_and_goal(x1,y1,theta1,goalx,goaly,tgd,xhigh)
quiver(x1,y1,cosd(theta1)*0.1*abs(xhigh),sind(theta1)*0.1*abs(xhigh),LineWidth=1,Marker="p",MarkerSize=4,AutoScaleFactor=0.9,MaxHeadSize=1,Color='#77AC30')
quiver(goalx,goaly,cosd(tgd)*0.1*abs(xhigh),sind(tgd)*0.1*abs(xhigh),LineWidth=1,Marker="h",MarkerSize=4,AutoScaleFactor=0.9,MaxHeadSize=1,Color='k')
end

