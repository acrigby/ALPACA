clear;

smoothed = importdata("C:\Users\aidan\OneDrive - UW-Madison\PhD\Dymola Output\SmoothedMatConvertedFile_FF.mat");
original = importdata("C:\Users\aidan\OneDrive - UW-Madison\PhD\Dymola Output\OriginalMatConvertedFile.mat");

T_o = original.Power(:,1)-10000;
T_o_days = (T_o/(3600*24));
T_o_min = (T_o/60);
T_s = smoothed.Power(:,1)-10000;
T_s_days = (T_s/(3600*24));
T_s_min = (T_s/60);

%TES Power
figure
tiledlayout(1,2)

nexttile
hold on
plot(T_o_min,original.SGOutTemp(:,2)-273.15,'LineWidth',1.5)
plot(T_s_min,smoothed.SGOutTemp(:,2)-273.15,'LineWidth',1.5)
% plot(T_hours,MainBOP(:,2)/1e6,'LineWidth',1.5)
% plot(T_hours,TESBOP(:,2)/1e6,'LineWidth',1.5)
hold off
xlim([-5,20])
%xticks(0:4:24)
xlabel('Time / min')
ylabel('Temperature / \circC')
set(gca, 'FontName', 'Times New Roman')
ylim([380,420])
legend('SG T_{out} Original','SG T_{out} with Feed-Forward' ,'NumColumns',2)
grid on
pbaspect([2.5 1 1])
saveas(gcf,'ICAPP_Figures/FeedForwardSGOutTemp.jpg')

%TES Power
nexttile
hold on
plot(T_o_min,smoothed.FF(:,2),'LineWidth',1.5)
% plot(T_hours,MainBOP(:,2)/1e6,'LineWidth',1.5)
% plot(T_hours,TESBOP(:,2)/1e6,'LineWidth',1.5)
hold off
xlim([-5,20])
%xticks(0:4:24)
xlabel('Time / min')
ylabel('Feed-Forward Signal / kg/s')
set(gca, 'FontName', 'Times New Roman')
ylim([-1,1])
legend('Feed-Forward Signal' ,'NumColumns',2)
grid on
pbaspect([2.5 1 1])
saveas(gcf,'ICAPP_Figures/FeedForwardSGOutTemp.jpg')

