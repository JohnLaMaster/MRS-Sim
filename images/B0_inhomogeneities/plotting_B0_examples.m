%%
load('/home/john/Documents/Research/In-Vivo-MRSI-Simulator/images/B0_inhomogeneities/CC_30ms_publication_dirty_B0_01/dataset_spectra_0.mat')
save_names = {'with_B0.eps', 'no_B0.eps'};
plotting = spectra;
ch = [1]; %#ok<NBRAK2> %[1, 7, 5];
xmn = 0.2; xmx = 4.2;
ans0 = find(ppm>=0.2); 
ans1 = find(ppm>=4.2); 
ind=[ans0(1), ans1(1)]; 
for n=1:length(ch)
    for i=1:2 
        mx = max(squeeze(max(max(abs(plotting(i,1,1,:)))))); 
        mn = min(squeeze(min(min(plotting(i,1,1,:)./mx)))); 
        [size(mx); size(mn)]; [mn;mx];
        if mn>=-0.05 
            mn = -0.05; 
        else
            mn = mn - 0.05; 
        end
        figure
        hold on
        s = size(plotting);
        for ii=3.1:0.02:3.3
            if rem(ii,1)==0 || abs(ii)<1*10^-6 || (ii>=0.99 && ii<=1.01)
                xline(ii,'LineStyle','--','Color',[0.4 0.4 0.4],'HandleVisibility','off')
            else
                xline(ii,'LineStyle',':','Color',[0.65 0.65 0.65],'HandleVisibility','off')
            end
        end
        
        plot(ppm,squeeze(plotting(i,1,1,:)./mx),'DisplayName','Raw')
        plot(ppm,squeeze(plotting(i,2,1,:)./mx),'r','DisplayName','Fit')
        hold off
        legend('FontSize', 12,'FontName','arial',Location="northwest")
        set(gca,'xdir','reverse')
        xlim([3.1,3.3])
        ylim([-0.05, 1.0])
        axis off
        exportgraphics(gca,strjoin({'/home/john/Documents/Research/In-Vivo-MRSI-Simulator/images/B0_inhomogeneities/',save_names{i}},''),'Resolution',800)
    end
end
