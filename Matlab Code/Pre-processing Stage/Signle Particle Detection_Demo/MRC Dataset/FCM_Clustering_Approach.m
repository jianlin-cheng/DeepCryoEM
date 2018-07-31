%% CryoEm Project ...
% AutoPicker: Fully Automated Single Particle Picking in Cryo-EM Images...
% FCM: Clustering Approach...
%==========================================================================
% Adil Al-Azzawi & Jianlin Cheng
% University of Missouri-Columbia 
% aaadn5@mail.missouri.edu
%==========================================================================
clc; 
disp('_________________________________________________________');
disp('                                                         ');
disp('   S  E  G  M  E  N  T  A  T  I  O  N   -  S  T  E  P    ');
disp('                   U S I N G   F C M                     ');
disp('_________________________________________________________');
disp('  ');
disp('   AutoPicker: Fully Automated Single Particle Picking   ');
disp('                   Cryo-EM Images                        ');
disp('_________________________________________________________');
disp(' ');
%
% MRI images reading and pre processing... 
CRYOEM_images_dir='C:\Users\Adil Al-Azzawi\Desktop\Protein Project\Single Particle Picking\MRC DATASET';
% code folder
code_dir='C:\Users\Adil Al-Azzawi\Desktop\Protein Project\Single Particle Picking\Perfomance Results';
% output folder
clust_output_dir='C:\Users\Adil Al-Azzawi\Desktop\Protein Project\Single Particle Picking\Expermental Results\FCM Clustering Images\';
%
k_means_time=zeros(1,20);
%change the directory to the skin cancer images...
cd(CRYOEM_images_dir);
D = dir('*.tif');
    for n = 1:numel(D)
        close all;
        cd(CRYOEM_images_dir);
        %% Step(1): Read the Cryo-EM Images...
        clc; 
        disp('_________________________________________________________');
        disp('                                                         ');
        disp('   S  E  G  M  E  N  T  A  T  I  O  N   -  S  T  E  P    ');
        disp('                   U S I N G   F C M                     ');
        disp('_________________________________________________________');
        disp(' ');
        fprintf('The CryoEM Image No. : %d\n',n');
        originalImage = imread(D(n).name);
        
        %% Step(2): Pre-Processing...
        % normalized the Cryo-EM
        Inormalized = double(originalImage)./double(max(originalImage(:)));
        % Image adjusment
        limit=stretchlim(Inormalized);
        ad=imadjust(Inormalized,[limit(1) limit(2)]);  
        subplot(3,3,1);imshow(ad);title('CTF Image Adjusment')
        % Image Restoration
        I = histeq(ad);
        K = wiener2(I,[3 3]);
        subplot(3,3,2); imshow(K);title('Restored Image');
        % Cryo-EM Histo-Equalization
        I = histeq(K);
        subplot(3,3,3); imshow(I);title('Cryo-EM Histo-Equalization')
        % Adaptive Cryo-EM Histo-Equal.
        g=adapthisteq(I,'clipLimit',.02,'Distribution','rayleigh');
        subplot(3,3,4);imshow(g);title('Adaptive Cryo-EM Histo-Equal.')
        % Adaptive Cryo-EM Histo-Equal.
        im=adapthisteq(g,'clipLimit',.99,'Distribution','rayleigh');
        subplot(3,3,5);imshow(im);title('Adaptive Histo-Equal.')
        % Gaudided Filtering
        im=imguidedfilter(im);im=imguidedfilter(im);
        im=imguidedfilter(im);im=imguidedfilter(im);
        im=imadjust(im);
        subplot(3,3,6);imshow(im);title('Gaudided Filtering')
        % Morphological Operation
        SE=strel('disk',5);J = imclose(im,SE);J2=imadjust(J,[.1,.9]);
        subplot(3,3,7);title('Post-processing Morphological Operation')
        
        %% Step(1): Cluster the Cryo-EM Images...
        % Clustering 
        cd(code_dir); 
        cluster_image = FCM_Clustering(J2);
        time=toc;
        FCM_means_time(n)=toc;
        fprintf('Time consuming time for doing MRI clustering is %f second \n',FCM_means_time(n));
        subplot(3,3,8);imshow(cluster_image);title('CryoEM Clustred Image')
        % Post Processing
        k=imfill(cluster,'holes');
        BW = imopen(k,SE);imshow(BW,[])
        c=bwareaopen(BW,100);
        subplot(3,3,9); imshow(c);title('Binary Mask Cleaning Image')
        %
        %% Save the cluster MRI Images ...
        imwrite(cluster_image, [clust_output_dir D(n).name],'jpg');         
        cd (CRYOEM_images_dir);
%         pause;
        close all;
    end
disp('All the CryoEM images have been done (Clustred using K-means) ...');
FCM_average_time=mean(FCM_means_time);
fprintf('The Mean Average time that K-Means needs to cluster all MRI images is : %f\n',FCM_average_time);