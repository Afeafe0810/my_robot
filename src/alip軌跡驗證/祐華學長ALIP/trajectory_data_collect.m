CX = cat(2,cx,cy,cz);
LX = cat(2,l_leg_x,l_leg_y,l_leg_z);
RX = cat(2,r_leg_x,r_leg_y,r_leg_z);

scatter3(CX(:,1),CX(:,2),CX(:,3),'filled','o','black');
hold on;
scatter3(LX(:,1),LX(:,2),LX(:,3),'filled','o','red');
scatter3(RX(:,1),RX(:,2),RX(:,3),'filled','o','blue');

csvFileName = 'CX_ref.csv';
writematrix(CX, csvFileName);

csvFileName = 'LX_ref.csv';
writematrix(LX, csvFileName);

csvFileName = 'RX_ref.csv';
writematrix(RX, csvFileName);

csvFileName = 'xc_ref.csv';
writematrix(xc, csvFileName);

csvFileName = 'yc_ref.csv';
writematrix(yc, csvFileName);

csvFileName = 'ly_ref.csv';
writematrix(ly, csvFileName);

csvFileName = 'lx_ref.csv';
writematrix(lx, csvFileName);