clc; close all; clear all;

[img,path]=uigetfile('*.jpg','Select an Image');
% open the directory box
str=strcat(path,img);

% read the MRI image from the spesific directory
MRC=imread(str);
% imshow(input_im);
% FILE_NAME='ALDH7A1_UF_A3__0001.tif';
% MRC=imread(FILE_NAME);
z=mat2gray(MRC);
figure;imshow(z);title('CryoEM Image')

originalImage=MRC;
Inormalized = double(originalImage)./double(max(originalImage(:)));
figure;imshow(Inormalized);title('Original Cryo-EM Image')

limit=stretchlim(Inormalized);
ad=imadjust(Inormalized,[limit(1) limit(2)]);  
figure;imshow(ad);title('CTF Image Adjusment')

I = histeq(ad);
K = wiener2(I,[3 3]);
% subplot(3,3,2); imshow(K);title('Restored Image');
I = histeq(K);
% subplot(3,3,3); 
figure;imshow(I);title('Cryo-EM Histo-Equalization')
% I1 = imcrop(I,[0.5 0.5 478 525]);
% figure;imshow(I1);title('Histo-Equalization')

g=adapthisteq(I,'clipLimit',.02,'Distribution','rayleigh');
% g1 = imcrop(g,[0.5 0.5 478 525]);
% figure;imshow(g1);title('Adaptive Histo-Equal.')
% subplot(3,3,4);
% imshow(g);title('Adaptive Cryo-EM Histo-Equal.')

im=adapthisteq(g,'clipLimit',.99,'Distribution','rayleigh');
% im1 = imcrop(im,[0.5 0.5 478 525]);
% figure;imshow(im1);title('Adaptive Histo-Equal.')
% subplot(3,3,5);
figure;imshow(im);title('Adaptive Cryo-EM Histo-Equal.')
% 
im=imguidedfilter(im);im=imguidedfilter(im);
im=imguidedfilter(im);im=imguidedfilter(im);
im=imadjust(im);
% im2 = imcrop(im,[0.5 0.5 478 525]);
% figure;imshow(im2);title('Gaudided Filtering')
% imwrite(im2,'Labled_Image.png');
% subplot(3,3,6);
figure;imshow(im);title('Gaudided Filtering')

% [J, rect] = imcrop(im);
% J = im2;
% figure;imshow(J);

SE=strel('disk',5);J = imclose(im,SE);J2=imadjust(J,[.1,.9]);
% J1 = imcrop(J,[0.5 0.5 478 525]);
% figure;imshow(J1);title('Morphological Operation')
% subplot(3,3,7);
figure;imshow(J2,[]);title('Post-processing Morphological Operation')

%% Clustering....
disp('_______________________________________________________________________');
disp('                                                                       ');
disp('        S I N G L E - P A R T I C L E - D E T E C T I O N ');
disp('                                                                       ');
disp('_______________________________________________________________________');
disp(' ');
disp('         1: Our Clustering Approach              ');
disp('         2: K-means Clustering Approach              ');
disp('         3: FCM Clustering Approach              ');
disp('         4: Exit ');
disp('_______________________________________________________________________');
disp(' ');
choice=input('Selct your choice : ');
tic;
if choice==1
% Our Approach
    [cluster] = Our_Clustering(J2);
    time=toc;
     fprintf(' Time consuming for Particle Detection using Our Approach is : %f\n', time);
     pause;
elseif choice==2
% K-Means
    [cluster] = Kmeans_Clustering_Approach(J2);
    time=toc;
     fprintf(' Time consuming for Particle Detection using K-means is : %f\n', time);
     pause;    
elseif choice==3
% FCM
    [cluster] = FCM_Clustering_Approach(J2);
    time=toc;
     fprintf(' Time consuming for Particle Detection using FCM is : %f\n', time);
     pause;    
end
%

%% Post-Processing...
% cluster_4=imbinarize(cluster);
% subplot(3,3,8);
figure; imshow(cluster);title('CryoEM Clustred Image')
%
k=imfill(cluster,'holes');
BW = imopen(k,SE);imshow(BW,[])
c=bwareaopen(BW,100);
% subplot(3,3,9);
figure; imshow(c);title('Binary Mask Cleaning Image')

binReg=regionprops(c,'All');
numb=size(binReg,1);
figure;imshowpair(c,ad,'blend');title(['Number of Detected Cells= ' num2str(numb)]); 

figure;imshow(z);title('CryoEM AutoPicker: Automated Single Particle Piking');
cell_str=regionprops(c,'All');

for k = 1 : length(cell_str)
  thisBB = cell_str(k).BoundingBox;
  rectangle('Position', [thisBB(1),thisBB(2),thisBB(3),thisBB(4)],...
  'EdgeColor','r','LineWidth',2 )
end
%


