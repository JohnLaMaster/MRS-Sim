%%
% load('/home/john/Documents/Research/Spectroscopy_Model/docs/MRS-Sim/test_sim/dataset_spectra_0.mat')
load('/home/john/Documents/Research/Spectroscopy_Model/docs/MRS-Sim/test_sim/final_day_clean_cc/dataset_spectra_0.mat')
plotting1 = fftshift(fft(spectra(:,:,1,:)+1j*spectra(:,:,2,:),[],4),4); %baselines; %spectra;
plotting(:,:,1,:) = real(plotting1);
plotting(:,:,2,:) = imag(plotting1);
% clear plotting1
% save_names = {'30ms_curated_sample0.eps';'30ms_curated_sample1.eps';'30ms_curated_sample2.eps';'30ms_curated_sample3.eps';
%     '30ms_curated_sample4.eps';'30ms_curated_sample5.eps'};
% plotting = spectra;
ch = [1]; %#ok<NBRAK2> %[1, 7, 5];
xmn = 0.2; xmx = 4.2;
ans0 = find(ppm>=0.2); 
ans1 = find(ppm>=4.2); 
ind=[ans0(1), ans1(1)]; 
cnt = 0;

phi1_ref = ppm - 4.65;
phi = params.phi1;
phi1 = phi1_ref .* deg2rad(params.phi1)';
PHI = reshape(phi1,100,1,8192);
clear phi1
phi1(:,1,:,:) = PHI;
phi1(:,2,:,:) = PHI;

% i=[4,2,7,8,10,12,13,14,16,17,18,19,22,35,37];
temp = plotting1 .* exp(1j*phi1);
plotting(:,:,1,:) = real(temp);
plotting(:,:,2,:) = imag(temp);

for n=1:length(ch)
%     n = ch(n);
    mx = [];
%     cnt=1;
    for i=[23,2,7,8,10,12,13,14,16,17,18,19,4,37,35]%31:40%21:30%[4,2,7,8,10,12,13,14,16,17,18,19]%1:20%[1,3,4,6,7,10]%1:10 
%         cnt=3; %3;
        %         phi1 = phi1_ref .* deg2rad(params.phi1);
%         mx = max(squeeze(max(max(abs(plotting(i,1,1,ind(1):ind(2))))))); 
%         mn = min(squeeze(min(min(plotting(i,1,1,ind(1):ind(2)))))); 
        mx = [mx; max(max(squeeze(max(max(abs(plotting(i,1,:,:)))))))]; 
        mn = min(min(squeeze(min(min(plotting(i,1,:,:)./mx(end)))))); 
        [size(mx(end)); size(mn)]; [mn;mx(end)];
        if mn>=-0.05 
            mn = -0.05; 
        else
            mn = mn - 0.05; 
        end
        figure
        hold on
        s = size(plotting);
    %     [size(ppm); size(squeeze(plotting(i,1,ii,1,:)))];
%         for ii=floor(min(ppm)*10)/10:0.2:ceil(max(ppm)*10)/10
         for ii=floor(min(ppm)):2:ceil(max(ppm)*10)/10
           if rem(ii,1)==0 || abs(ii)<1*10^-6 || (ii>=0.99 && ii<=1.01)
                xline(ii,'LineStyle','--','Color',[0.65 0.65 0.65],'HandleVisibility','off')
            else
                xline(ii,'LineStyle',':','Color',[0.65 0.65 0.65],'HandleVisibility','off')
            end
        end
        
%         yticks([])
    % %     size(squeeze(mean(spectra(i,1,:,:),3)))
    %     plot(ppm,squeeze(mean(spectra(i,1,:,1,:),3))'./mx)
        plot(ppm,squeeze(plotting(i,1,2,:)./mx(end)),'DisplayName','Raw - Imag','Color',[0.7,0.7,0.7])
        plot(ppm,squeeze(plotting(i,1,1,:)./mx(end)),'DisplayName','Raw - Real','Color',[0 0 0 ])%[0 0.4470 0.7410])
        plot(ppm,squeeze(plotting(i,2,1,:)./mx(end)),'r-','DisplayName','Fit')
%         plot(ppm,fliplr(squeeze(baselines(i,1,1,:)./mx)),'DisplayName','Baseline','color','#77AC30')
        plot(ppm,squeeze(baselines(i,1,2,:)./mx(end)),'DisplayName','Baseline','color','#77AC30')
        plot(ppm,squeeze(residual_water(i,1,1,:)./mx(end)),'DisplayName','Residual Water','Color',[0.47 0.25 0.80])
%         plot(ppm,squeeze(plotting(i,1,2,:)./mx),'DisplayName','Raw - Imag','Color',[0.7,0.7,0.7])
        hold off
        if cnt==0
            legend('Location',"northeast")%"northwest"
        end
        set(gca,'xdir','reverse','FontSize',14,'FontName','Times')
        set(gca,'YTickLabel',[]);
        set(gcf, "Position",[1351 673 570 422])
        set(gca,"Position",[75.1 47.42 441.75 343.93])
%         xlim([xmn, xmx])
        xlim([0,5])
%         xlim([min(ppm) max(ppm)])
        ylim([mn, 1.0])
%         axis off
        cnt = cnt + 1;
        name = sprintf('30ms_curated_sample{%d}.eps', cnt);
% %         exportgraphics(gca,strjoin({'/home/john/Documents/Repositories/MRS-Sim/images/30ms_curated_samples/',name},''),'Resolution',800)
%         exportgraphics(gca,strjoin({'/home/john/Documents/Repositories/MRS-Sim/images/30ms_curated_samples_no_phi1/',name},''),'Resolution',800)
    end
end
