%% PM Stages
load('/home/john/Documents/Research/Spectroscopy_Model/docs/MRS-Sim/test_sim/pm_stages/dataset_spectra_0.mat')
clear plotting1 plotting
plotting1 = fftshift(fft(spectra(:,:,1,:)+1j*spectra(:,:,2,:),[],4),4); %baselines; %spectra;
plotting(:,:,1,:) = real(plotting1);
plotting(:,:,2,:) = imag(plotting1);
clear plotting1
% plotting = spectra;
ch = [1]; %#ok<NBRAK2> %[1, 7, 5];
xmn = 0.2; xmx = 4.2;
ans0 = find(ppm>=0.2); 
ans1 = find(ppm>=4.2); 
ind=[ans0(1), ans1(1)]; 
cnt = 0;
for n=1:length(ch)
%     n = ch(n);
    mx = [];
%     for i=[1,13,4,12]
    for i=1:12%:-1:1
%         1:30%[16,1,4,7,8,10,12,13,14,17,18,19,26,37,38]%1:20%,20:40%[4,2,7,8,1IMPnRXEWpvr0,12,13,14,16,17,18,19]%1:20%[1,3,4,6,7,10]%1:10 
%         mx = max(squeeze(max(max(abs(plotting(i,1,1,ind(1):ind(2))))))); 
%         mn = min(squeeze(min(min(plotting(i,1,1,ind(1):ind(2)))))); 
%         [plotting(i,1,:,:), ]
        if i==1 || i==12
            ii=i;
        elseif i>1 && i<12
            ii=i-1;
        end
        mx = [mx; max(max(squeeze(max(max(abs(plotting(ii:i,1,:,:)))))))]; 
        mn = min(min(squeeze(min(min(plotting(i,1,:,:)./mx(end)))))); 
        [size(mx(end)); size(mn)]; [mn;mx(end)];
        if mn>=-0.05 
            mn = -0.05; 
        else
            mn = mn - 0.05; 
        end
        mult=2;
        figure('units','centimeters','position',[0,0,5.225*mult,1.8*mult]);
        hold on
        s = size(plotting);
    %     [size(ppm); size(squeeze(plotting(i,1,ii,1,:)))];
%         for ii=floor(min(ppm)*10)/10:0.2:ceil(max(ppm)*10)/10
         for ii=floor(min(ppm)):0.5:ceil(max(ppm)*10)/10
           if rem(ii,1)==0 || abs(ii)<1*10^-6 || (ii>=0.99 && ii<=1.01)
                xline(ii,'LineStyle','--','Color',[0.65 0.65 0.65],'HandleVisibility','off')
            else
                xline(ii,'LineStyle',':','Color',[0.65 0.65 0.65],'HandleVisibility','off')
            end
         end
         xline(0,'k','HandleVisibility','off')
         xline(5,'k','HandleVisibility','off')
        
%         yticks([])
    % %     size(squeeze(mean(spectra(i,1,:,:),3)))
    %     plot(ppm,squeeze(mean(spectra(i,1,:,1,:),3))'./mx)
        plot(ppm,squeeze(plotting(i,1,2,:)./mx(end)),'DisplayName','Raw - Imag','Color',[0.7,0.7,0.7])
        plot(ppm,squeeze(plotting(i,1,1,:)./mx(end)),'k','DisplayName','Raw - Real')
        if i>1 && i~=11 && i~=12 && i~=3 && i~=4 && i~=5% && i~=6
            plot(ppm,squeeze(plotting(i-1,2,1,:)./mx(end)*2),'r','DisplayName','Fit')
        elseif i==3
            plot(ppm,squeeze(plotting(i-1,2,1,:)./mx(end)),'r','DisplayName','Fit')
            plot(ppm,squeeze(plotting(i,1,1,:)./mx(end)),'k','DisplayName','Raw - Real')
        elseif i==4 
            plot(ppm,squeeze(plotting(i-1,2,1,:)./mx(end)),'r','DisplayName','Fit')
            plot(ppm,squeeze(residual_water(i,1,1,:)./mx(end)),'Color',[0.47 0.25 0.80])
        elseif i==5
            plot(ppm,squeeze(plotting(i-1,2,1,:)./mx(end)),'r','DisplayName','Fit')
            plot(ppm,squeeze(baselines(i,1,1,:)./mx(end)),'Color','#77AC30')
%         elseif i==6
%             mx_temp = squeeze(max(plotting(i-1,2,1,:)));
%             plot(ppm,squeeze(plotting(i-1,2,1,:)./mx_temp),'r','DisplayName','Fit')
        elseif i==12
             plot(ppm,squeeze(plotting(i,2,1,:)./mx(end)),'r','DisplayName','Fit')
             plot(ppm,squeeze(baselines(i,1,1,:)./mx(end)),'Color','#77AC30')
        end
%         plot(ppm,squeeze(plotting(i,1,1,:)./mx(end)),'DisplayName','Raw - Real','Color',[0 0.4470 0.7410])
%         plot(ppm,squeeze(plotting(i,2,1,:)./mx(end)),'r-','DisplayName','Fit')
%         plot(ppm,fliplr(squeeze(baselines(i,1,1,:)./mx)),'DisplayName','Baseline','color','#77AC30')
%         plot(ppm,squeeze(baselines(i,1,2,:)./mx(end)),'DisplayName','Baseline','color','#77AC30')
%         plot(ppm,squeeze(residual_water(i,1,1,:)./mx(end)),'DisplayName','Residual Water','Color',[0.47 0.25 0.80])
%         plot(ppm,squeeze(plotting(i,1,2,:)./mx),'DisplayName','Raw - Imag','Color',[0.7,0.7,0.7])
        hold off
%         if cnt==0
%             legend(Location="northeast")
%         end
        set(gca,'xdir','reverse','FontSize',14,'FontName','Times')
        set(gca,'YTickLabel',[],'YTick',[]);%,'YTicks',[]
        set(gca,'XTickLabel',[0,1,2,3,4,5],'XTick',[0,1,2,3,4,5])
%         if i<9
        set(gca,'XTickLabel',[],'XTick',[])
%         end
        box on
%         xlim([xmn, xmx])
        xlim([0,5])
%         xlim([min(ppm) max(ppm)])
        ylim([mn, 1.0])
%         axis off
        cnt = cnt + 1;
        name = sprintf('pm_stages_%d.eps', cnt);
        exportgraphics(gca,strjoin({'/home/john/Documents/Repositories/MRS-Sim/images/new_PM_stages/',name},''),'Resolution',800)
    end
end