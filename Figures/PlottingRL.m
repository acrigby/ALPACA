

load("C:/Users/localuser/Documents/Dymola/MatConvertedFile");

T = Power(:,1)-10000;
T_days = (T/(3600*24));
T_min = (T/60);


%TES Power
figure
hold on
plot(T_min,Power(:,2)/1e6,'LineWidth',1.5)
plot(T_min,Demand(:,2)/1e6,'LineWidth',1.5)
% plot(T_hours,MainBOP(:,2)/1e6,'LineWidth',1.5)
% plot(T_hours,TESBOP(:,2)/1e6,'LineWidth',1.5)
hold off
xlim([-5,60])
%xticks(0:4:24)
xlabel('Time / min')
ylabel('Power / MWe')
ylim([0,50])
legend('Eletrical Power', 'Demanded Power', 'NumColumns',2)
grid on
pbaspect([2.5 1 1])
saveas(gcf,'Figures/Power.jpg')

%TES Power
figure
hold on
plot(T_min,SGOutTemp(:,2)-273.15,'LineWidth',1.5)
% plot(T_hours,MainBOP(:,2)/1e6,'LineWidth',1.5)
% plot(T_hours,TESBOP(:,2)/1e6,'LineWidth',1.5)
hold off
xlim([-5,60])
%xticks(0:4:24)
xlabel('Time / min')
ylabel('Temperature / \circC')
ylim([380,420])
legend('Steam Generator Outlet Temperature', 'NumColumns',2)
grid on
pbaspect([2.5 1 1])
saveas(gcf,'Figures/SGOutTemp.jpg')

%TES Power
figure
hold on
plot(T_min,PumpMFlow(:,2),'LineWidth',1.5)
% plot(T_hours,MainBOP(:,2)/1e6,'LineWidth',1.5)
% plot(T_hours,TESBOP(:,2)/1e6,'LineWidth',1.5)
hold off
xlim([-5,60])
%xticks(0:4:24)
xlabel('Time / min')
ylabel('Mass Flow Rate / kg/s')
ylim([58,59])
legend('FWCP Mass Flow Rate', 'NumColumns',2)
grid on
pbaspect([2.5 1 1])
saveas(gcf,'Figures/FWCP_MFlow.jpg')

%TES Power
figure
hold on
plot(T_min,DLevel(:,2),'LineWidth',1.5)
% plot(T_hours,MainBOP(:,2)/1e6,'LineWidth',1.5)
% plot(T_hours,TESBOP(:,2)/1e6,'LineWidth',1.5)
hold off
xlim([-5,60])
%xticks(0:4:24)
xlabel('Time / min')
ylabel('Level / m')
ylim([2.8,3.2])
legend('Deaerator Water Level', 'NumColumns',2)
grid on
pbaspect([2.5 1 1])
saveas(gcf,'Figures/Level.jpg')

