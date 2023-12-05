%%
% load('/home/john/Documents/Research/In-Vivo-MRSI-Simulator/images/B0_inhomogeneities/CC_30ms_publication_dirty_B0_01/dataset_spectra_0.mat')
load('/home/john/Documents/Repositories/MRS-Sim/images/phase/phase1/dataset_spectra_0.mat')
save_names = {'no_phase.eps', 'zero-order.eps', 'first-order.eps'};
plotting = spectra;
ch = [1]; %#ok<NBRAK2> %[1, 7, 5];
xmn = 0.2; xmx = 4.2;
ans0 = find(ppm>=0.2); 
ans1 = find(ppm>=4.2); 
ind=[ans0(1), ans1(1)]; 
track = [];
cnt = 0;
for n=1:length(ch)
    for i=[2,4,6]
        mx = max(squeeze(max(max(abs(plotting(i,1,1,:)))))); 
        mn = min(squeeze(min(min(plotting(i,1,1,:)./mx)))); 
        [size(mx); size(mn)]; track = [track, [mn;mx]];
        if mn>=-0.05 
            mn = -0.05; 
        else
            mn = mn - 0.05; 
        end
        figure
        hold on
        s = size(plotting);
%         for ii=3.1:0.02:3.3
        for ii=0:0.2:5
            if rem(ii,1)==0 || abs(ii)<1*10^-6 || (ii>=0.99 && ii<=1.01)
                xline(ii,'LineStyle','--','Color',[0.4 0.4 0.4],'HandleVisibility','off')
            else
                xline(ii,'LineStyle',':','Color',[0.65 0.65 0.65],'HandleVisibility','off')
            end
        end
        
%         plot(ppm,squeeze(plotting(i,1,1,:)./mx),'Color',[0 0.4470 0.7410],'DisplayName','Real')
        plot(ppm,squeeze(plotting(i,1,1,:)./mx),'Color',[0 0 0],'DisplayName','Real')
%         plot(ppm,squeeze(plotting(i,1,1,:)./mx),'k','DisplayName','Real')
%         plot(ppm,squeeze(plotting(i,1,2,:)./mx),'r','DisplayName','Imaginary')
        plot(ppm,squeeze(plotting(i,1,2,:)./mx),'Color',[0.7,0.7,0.7],'DisplayName','Imaginary')
        set(gca,'Children',flipud(get(gca,'Children')))
        hold off
        cnt = cnt+1;
        if cnt==1
            legend('FontSize', 12,'FontName','arial',Location="northwest")
        end
        set(gca,'xdir','reverse')
        xlim([0,5])%[3.1,3.3])
        ylim([-0.6, 1.0])
        axis off
        exportgraphics(gca,strjoin({'/home/john/Documents/Repositories/MRS-Sim/images/phase/phase1/',save_names{cnt}},''),'Resolution',800)
    end
end
