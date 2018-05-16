%% CryoEm Pre-processing Step...
%==========================================================================
% Adil Al-Azzawi & Jianlin Cheng
% University of Missouri-Columbia 
% aaadn5@mail.missouri.edu
%==========================================================================
% Issue: 
% Sine our CryoEm dataset in raw format which have not been pre-processed 
% before, ijn this case those images have CTF problem.
%
% Tool description: 
% in this step we are trying to solve the Constrast Transfer Function 
% (CTF) issue based on the the constrast function to create an Adjust Contrast 
% tool. 
% The Adjust Contrast tool is an interactive contrast and brightness adjustment
% tool that you can use to adjust the black-to-white mapping used to display 
% a grayscale image. For more information about using the tool
%__________________________________________________________________________
clc;
disp('___________________________________________________________');
disp('                                                           ');
disp('D  E  E  P  -  C  R  Y  O  -  E  M  -  P  R  O  J  E  C  T ');
disp('         P R E - P R O C E S S I N G - S T E P             ');
disp('             Contrast Transfer Function (CTF)              ');
disp('___________________________________________________________');
disp(' ');
%
%% Step(1): Initalization... 
%CryoEm folder
CryoEM_images_dir='C:\Users\Adil Al-Azzawi\Desktop\Protein Project\CryoEM Dataset';
%code folder
code_dir='C:\Users\Adil Al-Azzawi\Desktop\Protein Project\Matlab Code\Pre-processing Stage';
%output folder
CryoEm_output_dir='C:\Users\Adil Al-Azzawi\Desktop\Protein Project\Pre_processed CryoEM\';
%
consuming_time=zeros(1,30);
%change the directory to the skin cancer images...
cd(CryoEM_images_dir);
D = dir('*.tif');
    for n = 1:numel(D)
        close all;
        fprintf('The CryoEm Image No. : %d\n',n');
        tic;
        % CryoEm reading
        originalImage = imread(D(n).name);
        subplot(2,2,1); imshow(originalImage);title('Original CryoEm Image')
        % Compute the original image histogram
        subplot(2,2,2); imhist(originalImage);title('Histogram of the Original Cryo-Image');
        %
        %% Step(2): CryoEm images Processing... 
        % Normalized the CryoEm
        normalized_CryoEm = double(originalImage)./double(max(originalImage(:)));
        subplot(2,2,3); imshow(normalized_CryoEm);title('Normalized CryoEm Image')
        % Compute the normalized image histogram
        subplot(2,2,4); imhist(normalized_CryoEm);title('Histogram of the Normalized Cryo-Image');
        % Deted the image level
        limit1=stretchlim(normalized_CryoEm);
        % Adjust the Image based on the limit
        Adjust_CryoEM=imadjust(normalized_CryoEm,[limit1(1) limit1(2)]);
        figure; subplot(3,2,1); imshow(Adjust_CryoEM);title('Adjust-Normalized CryoEm Image')
        subplot(3,2,2); imhist(Adjust_CryoEM);title('Histogram of the Adjust-Normalized CryoEm Image');
        % CryoEm restoration using Histogram Equalization and Wiener filter
        Restored_CryoEM = histeq(Adjust_CryoEM);
        Restored_CryoEM = wiener2(Restored_CryoEM,[3 3]);
        subplot(3,2,3); imshow(Restored_CryoEM);title('CryoEm Restoration');
        % Compute the CryoEm restoration histogram
        subplot(3,2,4); imhist(Restored_CryoEM);title('Histogram of CryoEm Restoration Image');
        % histogram adaptaion 
        Histo_adaptaion_CryoEM=adapthisteq(Restored_CryoEM,'clipLimit',.02,'Distribution','rayleigh');
        %
        im=adapthisteq(Histo_adaptaion_CryoEM,'clipLimit',.99,'Distribution','rayleigh');
        subplot(3,2,5);imshow(im)
        % Compute the CryoEm restoration histogram
        subplot(3,2,6); imhist(im);title('Histogram of CryoEm Restoration Image');
        % Save the processed CryoEm image ...
        imwrite(im, [CryoEm_output_dir D(n).name],'tif');   
        % Change the Code direction
        cd(code_dir);
        %
        %% Step(3): CryoEm Images Quality estimation... 
        % % Calculate the Peak Signal to Signal Noise Ratio (PSNR)
        [PSNR_value,~] = psnr(double(normalized_CryoEm), double(im)); 
        % Calculate the Mean Sequare Error Ratio (MSE) 
        [MSE_value]=MSE((normalized_CryoEm),(im));
        % Calculate the Signal to Noise Ratio (SNR)
        SNR_value = SNR(double(normalized_CryoEm), double(im));
        % Display the Results
        disp('________________________________________________________________');
        disp(' ');
        disp('    C R Y O - E M - I M A G E - Q U A L I T Y - M E A S U R E   ');
        disp('                     P S N R - and - M S E                      ');
        disp('________________________________________________________________');
        disp(' ');
        fprintf(' 1:Peak Signal to Noise Ratio (PSNR) is : %5.5f\n', abs(PSNR_value));
        fprintf(' 3:Mean Squared Error (MSE)          is : %5.8f \n', MSE_value);
        fprintf(' 5:Signal to Noise Ratio (SNR)       is : %5.5f dB \n', SNR_value);
        disp('________________________________________________________________');
        %
        % get the the Average time consuming
        consuming_time(n)=toc;
        PSNR_values(n)=abs(PSNR_value);
        MSE_values(n)=abs(MSE_value);
        SNR_values(n)=abs(SNR_value);
%         pause;
        cd(CryoEM_images_dir);
        close all;
    end
disp('All The CryoEm images pre-processing have been done ...');
%
figure;
plot(PSNR_values);
title('CryoEM Images PSNR Quality Estimation')
hold on
xlabel('No. of CryoEm Images')
ylabel('Peak Signal to Signal Noise Ratio (PSNR)');

figure;
plot(MSE_values);
title('CryoEM Images SNR Quality Estimation')
xlabel('No. of CryoEm Images')
ylabel('Mean Sequare Error Ratio (MSE)');

figure;
plot(SNR_values);
title('CryoEM Images MSE Quality Estimation')
xlabel('No. of CryoEm Images')
ylabel('Signal to Noise Ratio (SNR)');
%
average_time=mean(consuming_time);
fprintf('The Average Consuming time t is : %f\n',average_time);

