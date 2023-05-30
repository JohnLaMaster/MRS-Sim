%%
load('/home/john/Documents/Research/In-Vivo-MRSI-Simulator/dataset/8C_30ms_publication_04_final/dataset_spectra_0.mat')
plotting = spectra; %baselines; %spectra;
ch = [1, 7, 5];
save_names = ['8coil_w_phase_w_fshift_cropped.eps', '8coil_wo_phase_w_fshift_cropped.eps', '8coil_wo_phase_wo_fshift_cropped.eps'];
xmn = 0.2; xmx = 4.2;
ans0 = find(ppm>=0.2); 
ans1 = find(ppm>=4.2); 
ind=[ans0(1), ans1(1)]; 
for n=1:length(ch)
    n = ch(n);
    for i=1%:10 
        mx = max(squeeze(max(max(abs(spectra(i,1,:,:,ind(1):ind(2))))))); 
        mn = min(squeeze(min(min(spectra(i,1,:,:,ind(1):ind(2)))))); 
        if mn>-0.05 
            mn = -0.05; 
        else
            mn = mn - 0.05; 
        end
        figure
        hold on
        s = size(plotting);
        for ii=1:s(3)
            plot(ppm,squeeze(plotting(i,n,ii,1,:))'./mx+1*ii)
        end
        for i=xmn:0.05:xmx
            xline(i,'LineStyle',':','Color',[0.65 0.65 0.65])
        end
        xline([1 2 3 4],'LineStyle','--','Color',[0.4 0.4 0.4])
        yticks([])
        hold off
        set(gca,'xdir','reverse')
        xlim([2.8, 3.4])%xmx])%1.6, xmx])
        axis off
        exportgraphics(gca,'/home/john/Documents/Research/In-Vivo-MRSI-Simulator/images/transients/'+save_name(n),'Resolution',800)
    end
end
% exportgraphics(gca,'/home/john/Documents/Research/In-Vivo-MRSI-Simulator/images/transients/8coil_wo_phase_wo_fshift_cropped.eps','Resolution',800)
% exportgraphics(gca,'/home/john/Documents/Research/In-Vivo-MRSI-Simulator/images/transients/8coil_wo_phase_w_fshift_cropped.eps','Resolution',800)
% exportgraphics(gca,'/home/john/Documents/Research/In-Vivo-MRSI-Simulator/images/transients/8coil_w_phase_w_fshift_cropped.eps','Resolution',800)

