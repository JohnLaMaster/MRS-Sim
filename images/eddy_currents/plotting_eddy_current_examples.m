%%
load('/home/john/Documents/Research/In-Vivo-MRSI-Simulator/dataset/CC_30ms_publication_dirty_ecc/dataset_spectra_0.mat')
save_names = {'ec=1'; 'ec=3'; 'ec=5'};
plotting = spectra;
ch = [1 3 5]; %#ok<NBRAK2> %[1, 7, 5];
xmn = 0.2; xmx = 4.2;
ans0 = find(ppm>=0.2); 
ans1 = find(ppm>=4.2); 
ind=[ans0(1), ans1(1)]; 
for n=1:length(ch)
    m = ch(n);
    for i=1%0:1 
        mx = max(squeeze(max(max(abs(plotting(n:n+1,1,1,:)))))); 
        figure
        hold on
        s = size(plotting);
        for ii=0.2:0.1:4.2
            if rem(ii,1)==0 || abs(ii)<1*10^-6 || (ii>=0.99 && ii<=1.01)
                xline(ii,'LineStyle','--','Color',[0.4 0.4 0.4],'HandleVisibility','off')
            else
                xline(ii,'LineStyle',':','Color',[0.65 0.65 0.65],'HandleVisibility','off')
            end
        end
        
%         plot(ppm,squeeze(plotting(m+1,1,1,:)./mx),'Color',[0.4 0.4 0.4],'DisplayName','With EC')
        plot(ppm,squeeze(plotting(m+1,1,1,:)./mx),'Color',[0.5 0.5 0.5],'DisplayName','With EC')
%         plot(ppm,squeeze(plotting(m+1,1,1,:)./mx),'r','DisplayName','With EC')
%         plot(ppm,squeeze(plotting(m,1,1,:)./mx),'Color',[0 0.4470 0.7410],'DisplayName','Without EC')
        plot(ppm,squeeze(plotting(m,1,1,:)./mx),'Color',[0 0 0],'DisplayName','Without EC')
        hold off
        if n==1
            legend('FontSize', 12,'FontName','arial',Location="northwest")
        end
        set(gca,'xdir','reverse')
        xlim([1.8 4.2])
        ylim([-0.05, 1.0])
        axis off
        name = strjoin({'/home/john/Documents/Repositories/MRS-Sim/images/eddy_currents/',save_names{n},'.eps'},'');
        exportgraphics(gca,name,'Resolution',800)

    end
end
