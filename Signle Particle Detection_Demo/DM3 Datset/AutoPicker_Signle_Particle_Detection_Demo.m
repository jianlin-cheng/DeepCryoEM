clc; close all; clear all;
%% Load the hdf5 file in MATLAB
FILE_NAME='ALDH7A1_NAD_UF_B5_0035.tif';
originalImage=imread(FILE_NAME);
imshow(originalImage); title('Original Cryo-Image');
figure; imhist(originalImage);title('Histogram of the Original Cryo-Image');
%
%% Pre-processing Part
Inormalized = double(originalImage)./double(max(originalImage(:)));
figure;imshow(Inormalized); title('Cryo-Image Normalized');
figure; imhist(Inormalized);title('Histogram of the Normalized Cryo-Image');
%
% imwrite(Inormalized,'DM3.tif');
I = histeq(Inormalized);
K = wiener2(I,[5 5]);
figure; imshow(K); title('Restored Cryo-Image');
figure; imhist(K);title('Histogram of the Restored Cryo-Image');
% imwrite(K,'Test.tif');
%
I = histeq(K);
figure; imshow(I); title('Histogram Equlaizer of Cryo-Image');
figure; imhist(I);title('Histogram of the Equalization Cryo-Image');
%
g=adapthisteq(I,'clipLimit',.02,'Distribution','rayleigh');
figure; imshow(I); title('Adaptive Histogram Equlaizer Cryo-Image');
figure; imhist(I);title('Histogram of the Equalization Cryo-Image');
%
im=adapthisteq(g,'clipLimit',.99,'Distribution','rayleigh');
figure; imhist(im);title('Histogram of the Adaptive Cryo-EM Histo-Equal.');
figure;imshow(im);title('Adaptive Cryo-EM Histo-Equal.')
% 
% im=imguidedfilter(im);im=imguidedfilter(im);
% im=imguidedfilter(im);im=imguidedfilter(im);
% im=imadjust(im);
figure;imshow(im);title('Gaudided Filtering')
figure; imhist(im);title('Histogram of the Gaudided Filtering');
% %
% SE=strel('disk',5);J = imclose(im,SE);J2=imadjust(J,[.5,.9]);
% figure;imshow(J2,[]);title('Post-processing Morphological Operation')
% figure; imhist(J2);title('Histogram of the ost-processing Morphological Operation');
% imwrite(im,'DM3_tested.tif');
imcl=imopen(im,strel('disk',1));
imcl=imopen(imcl,strel('disk',1));
imcl=imopen(imcl,strel('disk',1));
imcl=imopen(imcl,strel('disk',1));
imcl=imopen(imcl,strel('disk',1));
figure; imshow(imcl);
J2=imcl;
%
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
% choice=input('Selct your choice : ');
% if choice==1
% Our Approach
    tic;
    [cluster1] = Our_Clustering1(J2);
    time1=toc;
     fprintf(' Time consuming for Particle Detection using Our Approach is : %f\n', time1);
     pause;
% elseif choice==2
% K-Means
    tic;
    [cluster2] = K_means_Clustering(J2);
    time2=toc;
     fprintf(' Time consuming for Particle Detection using K-means is : %f\n', time2);
     pause;    
% elseif choice==3
% FCM
     tic;
    [cluster3] = FCM_Clustering(J2);
    time3=toc;
     fprintf(' Time consuming for Particle Detection using FCM is : %f\n', time3);
     pause;    
% end
%
%% Post-Processing...
% 
c1=cluster1;
binReg1=regionprops(c1,'All');
numb1=size(binReg1,1);
%
c2=cluster2;
binReg2=regionprops(c2,'All');
numb2=size(binReg2,1);
%
c3=cluster3;
binReg3=regionprops(c3,'All');
numb3=size(binReg3,1);
%
figure;imshowpair(c1,im,'blend');title(['Number of Detected Cells= ' num2str(numb1)]); 
% figure;imshow(z);title('CryoEM AutoPicker: Automated Single Particle Piking');
% cell_str=regionprops(c,'All');
% 
% for k = 1 : length(cell_str)
%   thisBB = cell_str(k).BoundingBox;
%   rectangle('Position', [thisBB(1),thisBB(2),thisBB(3),thisBB(4)],...
%   'EdgeColor','r','LineWidth',2 )
% end
%
cell_str1=regionprops(c1,'All');
cell_tbl1=struct2table(cell_str1);

cell_str2=regionprops(c2,'All');
cell_tbl2=struct2table(cell_str2);

cell_str3=regionprops(c3,'All');
cell_tbl3=struct2table(cell_str3);

% median(cell_tbl.FilledArea);

p1=prctile(cell_tbl1.FilledArea,[0 100]);
p2=prctile(cell_tbl2.FilledArea,[0 100]);
p3=prctile(cell_tbl3.FilledArea,[0 100]);

% idxLowCounts = cell_tbl.FilledArea < median(cell_tbl.FilledArea);
idxLowCounts1 = cell_tbl1.FilledArea >= p1(1);
idxLowCounts2 = cell_tbl2.FilledArea >= p2(1);
idxLowCounts3 = cell_tbl3.FilledArea >= p3(1);

%
cell_small1 = cell_tbl1(idxLowCounts1,:);
cell_BB1=cell_small1.BoundingBox;
w1=round(mean(cell_BB1(:,3)));
h1=round(mean(cell_BB1(:,4)));
%
[img1,path]=uigetfile('*.tiff','Select a MRI Brain Tumor Image');
% open the directory box
str=strcat(path,img1);
% read the MRI image from the spesific directory
Cryp_EM_Image=imread(str);
figure, imshow(Cryp_EM_Image);title('AutoCryoPicker:Fully Automated Single Particle Picking in Cryo-EM Images'); 
imwrite(Inormalized,'DM3.tif');
for k = 1 : length(cell_BB1)
  thisBB1 = cell_BB1(k,:);
  rectangle('Position', [thisBB1(1)-10,thisBB1(2)-10,w1*2,h1*2],...
  'EdgeColor','r','LineWidth',2 )
end
%
cell_small2 = cell_tbl2(idxLowCounts2,:);
cell_BB2=cell_small2.BoundingBox;
w2=round(mean(cell_BB2(:,3)));
h2=round(mean(cell_BB2(:,4)));
%
figure, imshow(Cryp_EM_Image);title('AutoCryoPicker:Fully Automated Single Particle Picking in Cryo-EM Images'); 
%imwrite(Inormalized,'DM3.tif');
for k = 1 : length(cell_BB2)
  thisBB2 = cell_BB2(k,:);
  rectangle('Position', [thisBB2(1)-10,thisBB2(2)-10,w2*1.5,h2*1.5],...
  'EdgeColor','k','LineWidth',2)
end

%
cell_small3 = cell_tbl3(idxLowCounts3,:);
cell_BB3=cell_small3.BoundingBox;
w3=round(mean(cell_BB3(:,3)));
h3=round(mean(cell_BB3(:,4)));
%
figure, imshow(Cryp_EM_Image);title('AutoCryoPicker:Fully Automated Single Particle Picking in Cryo-EM Images'); 
%imwrite(Inormalized,'DM3.tif');
for k = 1 : length(cell_BB3)
  thisBB3 = cell_BB3(k,:);
  rectangle('Position', [thisBB3(1)-10,thisBB3(2)-10,w3*1.5,h3*1.5],...
  'EdgeColor','b','LineWidth',2 )
end


