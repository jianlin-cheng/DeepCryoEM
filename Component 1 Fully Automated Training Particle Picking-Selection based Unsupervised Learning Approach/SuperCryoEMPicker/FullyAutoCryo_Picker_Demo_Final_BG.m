clc; close all; clear all;

[img,path]=uigetfile('*.tiff','Select any cryo-EM Image...');
% open the directory box
str=strcat(path,img);

% read the MRI image from the spesific directory
originalImage1=imread(str);
originalImage2=originalImage1;
originalImage1=rgb2gray(originalImage1);
% originalImage1 = imcomplement(originalImage1);
originalImage = imresize(originalImage1,.5);
cryo_EM_image = imresize(originalImage2,.5);
% originalImage=rgb2gray(originalImage);originalImage = imresize(originalImage,.5);
imwrite(originalImage,'originalImage.png');
figure;imshow(originalImage); title('Original Cryo-Image');
figure; imhist(originalImage);title('Histogram of the Original Cryo-Image');
orginal_I = imcrop(originalImage,[1030.875 727.625 92.75 79.25]);
imwrite(orginal_I,'originalImage_cropped.png');
%
%% Pre-processing Part
% Image normalization...
z=mat2gray(originalImage);
figure;imshow(z);title('Normalized cryo-EM Image')
figure; imhist(z);title('Histogram of the Cryo-Image');
imwrite(z,'Normalized.png');
Normalized_I = imcrop(z,[[1030.875 727.625 92.75 79.25]]);
imwrite(Normalized_I,'Normalized_cropped.png');

% Contrast Enhancement Correction
Inormalized=z;
limit=stretchlim(Inormalized);
CEC_Image_Adjusment=imadjust(Inormalized,[limit(1) limit(2)]);  
figure;imshow(CEC_Image_Adjusment);title('CEC Image Adjusment')
figure; imhist(CEC_Image_Adjusment);title('Histogram of the Cryo-Image');
imwrite(CEC_Image_Adjusment,'CEC.png');
CEC_Image_Adjusment_I = imcrop(CEC_Image_Adjusment,[1030.875 727.625 92.75 79.25]);
imwrite(CEC_Image_Adjusment_I,'CEC_cropped.png');

% Hostogram Equalization
Cryo_EM_Histogram_Equalization = histeq(CEC_Image_Adjusment);
figure; imshow(Cryo_EM_Histogram_Equalization); title('Cryo-EM Histogram Equalization');
figure; imhist(Cryo_EM_Histogram_Equalization);title('Histogram of the Cryo-Image');
imwrite(Cryo_EM_Histogram_Equalization,'HE.png');
Cryo_EM_Histogram_Equalization_I = imcrop(Cryo_EM_Histogram_Equalization,[1030.875 727.625 92.75 79.25]);
imwrite(Cryo_EM_Histogram_Equalization_I,'HE_cropped.png');

% Cryo-EM Restoration
Cryo_EM_Restoration = wiener2(Cryo_EM_Histogram_Equalization,[5 5]);
figure; imshow(Cryo_EM_Restoration); title('Cryo-EM Restoration');
figure; imhist(Cryo_EM_Restoration);title('Histogram of the Restored Cryo-Image');
imwrite(Cryo_EM_Restoration,'Restored.png');
Cryo_EM_Restoration_I = imcrop(Cryo_EM_Restoration,[1030.875 727.625 92.75 79.25]);
imwrite(Cryo_EM_Restoration_I,'Restored_cropped.png');

% Adaptive Histogram Equlaizer Cryo-Image
Adaptive_Histogram_Equlaizer = histeq(Cryo_EM_Restoration);
Adaptive_Histogram_Equlaizer=adapthisteq(Adaptive_Histogram_Equlaizer,'clipLimit',.02,'Distribution','rayleigh');
Adaptive_Histogram_Equlaizer=adapthisteq(Adaptive_Histogram_Equlaizer,'clipLimit',.99,'Distribution','rayleigh');
figure;imshow(Adaptive_Histogram_Equlaizer);title('Adaptive Cryo-EM Histo-Equal.')
figure; imhist(Adaptive_Histogram_Equlaizer);title('Histogram of the Adaptive Cryo-EM Histo-Equal.');
imwrite(Adaptive_Histogram_Equlaizer,'Adaptive_Histogram_Equlaizer.png');
Adaptive_Histogram_Equlaizer_I = imcrop(Adaptive_Histogram_Equlaizer,[1030.875 727.625 92.75 79.25]);
imwrite(Adaptive_Histogram_Equlaizer_I,'Adaptive_Histogram_Equlaizer_cropped.png');

% Gaudided Filtering
Gaudided_Filtering=imguidedfilter(Adaptive_Histogram_Equlaizer);
Gaudided_Filtering=imguidedfilter(Gaudided_Filtering);
Gaudided_Filtering=imguidedfilter(Gaudided_Filtering);
Gaudided_Filtering=imguidedfilter(Gaudided_Filtering);
Gaudided_Filtering=imadjust(Gaudided_Filtering);
figure;imshow(Gaudided_Filtering);title('Gaudided Filtering')
figure; imhist(Gaudided_Filtering);title('Histogram of the Gaudided Filtering');
imwrite(Gaudided_Filtering,'Gaudided.png');
Gaudided_Filtering_I = imcrop(Gaudided_Filtering,[1030.875 727.625 92.75 79.25]);
imwrite(Gaudided_Filtering_I,'Gaudided_Filtering_cropped.png');

% Morphological Image Operation
Morphological_Image=imopen(Gaudided_Filtering,strel('disk',3));
Morphological_Image=imopen(Morphological_Image,strel('disk',1));
Morphological_Image=imopen(Morphological_Image,strel('disk',1));
Morphological_Image=imopen(Morphological_Image,strel('disk',1));
Morphological_Image=imopen(Morphological_Image,strel('disk',1));
figure;imshow(Morphological_Image);title('Morphological Image Operation')
figure; imhist(Morphological_Image);title('Histogram of Morphological Image Operation');
imwrite(Morphological_Image,'Morphological.png');
Morphological_Image_I = imcrop(Morphological_Image,[1030.875 727.625 92.75 79.25]);
imwrite(Morphological_Image_I,'Morphological_cropped.png');

%% ICB Clustering ....... (Partricles Clustering Stage)
tic;
[Morphological_Image1] = ICB_Process(originalImage2);
[cluster_ICB] = ICB_Clustering_B(Morphological_Image1);
time1=toc;
fprintf(' Time consuming for Particle Detection using ICB Approach is : %f\n', time1);
pause;
%
L = logical(cluster_ICB);
cluster_ICB2 =im2double( bwareafilt(L,[10 1000]));
imwrite(cluster_ICB2,'ICB_clustering_after_cleaning.png');
%
f=cluster_ICB2;
cluster2=bwareaopen(f,0);
SE1=strel('disk',1);
cluster3=imerode(cluster2,SE1);
SE2=strel('disk',1);
cluster4=imdilate(cluster3,SE2);
figure;imshow(cluster4);title('ICB Clustering After Cleaning');
%
cell_str1=regionprops(cluster4,'All');
cell_tbl1=struct2table(cell_str1);
p1=prctile(cell_tbl1.FilledArea,[30 100]);
idxLowCounts1 = cell_tbl1.FilledArea >= p1(1);
cell_small1 = cell_tbl1(idxLowCounts1,:);
cell_BB1=cell_small1.Centroid;
figure, imshow(cryo_EM_image);title('AutoCryoPicker:Fully Automated Single Particle Picking in Cryo-EM Images (ICB)'); 
% Ilabel = bwlabel(cell_small1);
%         stat = regionprops(cell_small1,'centroid');  
for k = 1: length(cell_BB1)
    x1=round(cell_BB1(k,1));
    x2=round(cell_BB1(k,2));
  rectangle('Position', [x1-30 x2-30 60 60],...
  'EdgeColor','g','LineWidth',1 )
end


%% K-means Clustering...... (Partricles Clustering Stage)
tic;
[cluster_Kmeans] = K_means_Clustering_B(Morphological_Image,1);
imwrite(cluster_Kmeans,'k_means_clustering_before_cleaning.png');
time2=toc;
fprintf(' Time consuming for Particle Detection using K-means is : %f\n', time2);
pause; 
%
cluster_Kmeans2 =cluster_Kmeans;
imwrite(cluster_Kmeans2,'K_means_clustering_after_cleaning.png');
%
f=cluster_Kmeans2;
cluster2=bwareaopen(f,0);
SE1=strel('disk',1);
cluster3=imerode(cluster2,SE1);
SE2=strel('disk',1);
cluster4=imdilate(cluster3,SE2);
figure;imshow(cluster4);title('K-meaqns Clustering After Cleaning');
%
cell_str1=regionprops(cluster4,'All');
cell_tbl1=struct2table(cell_str1);
p1=prctile(cell_tbl1.FilledArea,[30 100]);
idxLowCounts1 = cell_tbl1.FilledArea >= p1(1);
cell_small1 = cell_tbl1(idxLowCounts1,:);
cell_BB1=cell_small1.Centroid;
figure, imshow(cryo_EM_image);title('AutoCryoPicker:Fully Automated Single Particle Picking in Cryo-EM Images (k-means)'); 
% Ilabel = bwlabel(cell_small1);
%         stat = regionprops(cell_small1,'centroid');  
for k = 1: length(cell_BB1)
    x1=round(cell_BB1(k,1));
    x2=round(cell_BB1(k,2));
  rectangle('Position', [x1-30 x2-30 60 60],...
  'EdgeColor','b','LineWidth',1 )
end

%% FCM Cryo-EM image Clustering... (Partricles Clustering Stage)
tic;
[cluster_FCM] = FCM_Clustering_B(Morphological_Image);
imwrite(cluster_FCM,'FCM_clustering.png');
time3=toc;
fprintf(' Time consuming for Particle Detection using FCM is : %f\n', time3);
pause;   
%
L = logical(cluster_FCM);
cluster_FCM2 =im2double( bwareafilt(L,[10 1000]));
imwrite(cluster_FCM2,'FCM_clustering_after_cleaning.png');
%
f=cluster_FCM2;
cluster2=bwareaopen(f,0);
figure;imshow(cluster2);
SE1=strel('disk',1);
cluster3=imerode(cluster2,SE1);
SE2=strel('disk',1);
cluster4=imdilate(cluster3,SE2);
figure;imshow(cluster4);title('FCM Clustering After Cleaning');
%
cell_str1=regionprops(cluster4,'All');
cell_tbl1=struct2table(cell_str1);
p1=prctile(cell_tbl1.FilledArea,[30 100]);
idxLowCounts1 = cell_tbl1.FilledArea >= p1(1);
cell_small1 = cell_tbl1(idxLowCounts1,:);
cell_BB1=cell_small1.Centroid;
figure, imshow(cryo_EM_image);title('AutoCryoPicker:Fully Automated Single Particle Picking in Cryo-EM Images (FCM)'); 
% Ilabel = bwlabel(cell_small1);
%         stat = regionprops(cell_small1,'centroid');  
for k = 1: length(cell_BB1)
    x1=round(cell_BB1(k,1));
    x2=round(cell_BB1(k,2));
  rectangle('Position', [x1-30 x2-30 60 60],...
  'EdgeColor','c','LineWidth',1 )
end

%% Region Based Image clustering...
Temp=Morphological_Image;
% Norm Raw Calculated
norm_RAW = double(Temp - min(Gaudided_Filtering(:)))/double(max(Gaudided_Filtering(:))-min(Gaudided_Filtering(:)));
% douple the image
A=double(Temp);
N=6000;
[L,NumLabels] = superpixels(A,N);
superpixel_cryo_EM = zeros(size(A),'like',A);
idx = label2idx(L);
numRows = size(A,1);
numCols = size(A,2);
for labelVal = 1:NumLabels
    redIdx = idx{labelVal};
    superpixel_cryo_EM(redIdx) = mean(A(redIdx));
end    
%
figure; imshow(superpixel_cryo_EM,[]);title(['Superpixel cryo-EM image - Number of Labels =' num2str(NumLabels)])
figure; imhist(superpixel_cryo_EM);title('Histogram of Superpixel cryo-EM image');

%% SP-ICB based Superpixle Resolution Cryo-EM image Clustering... (Partricles Clustering Stage)
tic;
[cluster_SP_ICB1] = Our_Clustering(superpixel_cryo_EM);
time4=toc;
fprintf(' Time consuming for Particle Detection using SP-ICB Approach is : %f\n', time4);
pause;
%
figure;imshow(cluster_SP_ICB1);title('Cleaned Cryo-EM Binary Mask Image');
L = logical(cluster_SP_ICB1);
%
% cluster_SP_ICB2 =im2double( bwareafilt(L,900,'smallest'));
cluster_SP_ICB2 =im2double( bwareafilt(L,[10 1000]));
figure;imshow(cluster_SP_ICB2);title('Objects with at least 100 pixels');
imwrite(cluster_SP_ICB2,'SP_ICB_clustering.png');
%
f=cluster_SP_ICB2;
cluster2=bwareaopen(f,0);
figure;imshow(cluster2);
SE1=strel('disk',1);
cluster3=imerode(cluster2,SE1);
SE2=strel('disk',1);
cluster4=imdilate(cluster3,SE2);
figure;imshow(cluster4);
%
cell_str1=regionprops(cluster4,'All');
cell_tbl1=struct2table(cell_str1);
p1=prctile(cell_tbl1.FilledArea,[20 100]);
idxLowCounts1 = cell_tbl1.FilledArea >= p1(1);
cell_small1 = cell_tbl1(idxLowCounts1,:);
cell_BB1=cell_small1.Centroid;
figure, imshow(originalImage);title('AutoCryoPicker:Fully Automated Single Particle Picking in Cryo-EM Images (ICB)'); 
% Ilabel = bwlabel(cell_small1);
%         stat = regionprops(cell_small1,'centroid');  
for k = 1: length(cell_BB1)
    x1=round(cell_BB1(k,1));
    x2=round(cell_BB1(k,2));
  rectangle('Position', [x1-30 x2-30 60 60],...
  'EdgeColor','k','LineWidth',1 )
end

%% SP-K-Means Cryo-EM image Clustering... (Partricles Clustering Stage)
tic;
[cluster_SP_Kmeans] = K_means_Clustering_BG(superpixel_cryo_EM);
time5=toc;
fprintf(' Time consuming for Particle Detection using SP-K-means is : %f\n', time5);
pause; 
%
figure;imshow(cluster_SP_Kmeans);title('Cleaned Cryo-EM Binary Mask Image');
L = logical(cluster_SP_Kmeans);
%
cluster_SP_Kmeans2 =im2double( bwareafilt(L,[10 1000]));
figure;imshow(cluster_SP_Kmeans2);title('Cleaned Cryo-EM Binary Mask Image');
imwrite(cluster_SP_Kmeans2,'cluster_SP_Kmeans.png');
%
f=cluster_SP_Kmeans2;
cluster2=bwareaopen(f,0);
figure;imshow(cluster2);
SE1=strel('disk',1);
cluster3=imerode(cluster2,SE1);
SE2=strel('disk',1);
cluster4=imdilate(cluster3,SE2);
figure;imshow(cluster4);
%
cell_str1=regionprops(cluster4,'All');
cell_tbl1=struct2table(cell_str1);
p1=prctile(cell_tbl1.FilledArea,[10 100]);
idxLowCounts1 = cell_tbl1.FilledArea >= p1(1);
cell_small1 = cell_tbl1(idxLowCounts1,:);
cell_BB1=cell_small1.Centroid;
figure, imshow(originalImage);title('AutoCryoPicker:Fully Automated Single Particle Picking in Cryo-EM Images (SP-K-means)'); 
% Ilabel = bwlabel(cell_small1);
%         stat = regionprops(cell_small1,'centroid');  
for k = 1: length(cell_BB1)
    x1=round(cell_BB1(k,1));
    x2=round(cell_BB1(k,2));
  rectangle('Position', [x1-30 x2-30 60 60],...
  'EdgeColor','y','LineWidth',1 )
end

%% SP-FCM Cryo-EM image Clustering... (Partricles Clustering Stage)
tic;
[cluster_SP_FCM] = FCM_Clustering(superpixel_cryo_EM);
time6=toc;
fprintf(' Time consuming for Particle Detection using SP-K-means is : %f\n', time6);
pause; 
%
figure;imshow(cluster_SP_FCM);title('Cleaned Cryo-EM Binary Mask Image');
L = logical(cluster_SP_FCM);
%
cluster_SP_FCM2 =im2double( bwareafilt(L,[10 1000]));
figure;imshow(cluster_SP_FCM2);title('Cleaned Cryo-EM Binary Mask Image');
imwrite(cluster_SP_FCM2,'cluster_SP_FCM.png');
%
f=cluster_SP_FCM2;
cluster2=bwareaopen(f,0);
figure;imshow(cluster2);
SE1=strel('disk',1);
cluster3=imerode(cluster2,SE1);
SE2=strel('disk',1);
cluster4=imdilate(cluster3,SE2);
figure;imshow(cluster4);
%
cell_str1=regionprops(cluster4,'All');
cell_tbl1=struct2table(cell_str1);
p1=prctile(cell_tbl1.FilledArea,[10 100]);
idxLowCounts1 = cell_tbl1.FilledArea >= p1(1);
cell_small1 = cell_tbl1(idxLowCounts1,:);
cell_BB1=cell_small1.Centroid;
figure, imshow(originalImage);title('AutoCryoPicker:Fully Automated Single Particle Picking in Cryo-EM Images (SP-FCM)'); 
% Ilabel = bwlabel(cell_small1);
%         stat = regionprops(cell_small1,'centroid');  
for k = 1: length(cell_BB1)
    x1=round(cell_BB1(k,1));
    x2=round(cell_BB1(k,2));
  rectangle('Position', [x1-30 x2-30 60 60],...
  'EdgeColor','c','LineWidth',1 )
end

%% Fast FRCM K-Means Cryo-EM image Clustering... (Partricles Clustering Stage)
cluster=5;
Max_w_size=3; Min_w_size=9;
tic;
[center1,U1,~,t1]=FRFCM(double(superpixel_cryo_EM),cluster,Max_w_size,Min_w_size);
time7=toc;
fprintf(' Time consuming for Particle Detection using SP-FRFCM is : %f\n', time7);
pause; 

figure
for i=1:cluster
    imgfi=reshape(U1(i,:,:),size(superpixel_cryo_EM,1),size(superpixel_cryo_EM,2));
    subplot(2,3,i+1); imshow(imgfi,[])
    title(['Index No: ' int2str(i)])
end

cluster_1=imbinarize(reshape(U1(1,:,:),size(originalImage,1),size(originalImage,2)));
Total_White_Pixels1 = nnz(cluster_1);
% figure;imshow(cluster_1);

cluster_2=imbinarize(reshape(U1(2,:,:),size(originalImage,1),size(originalImage,2)));
Total_White_Pixels2 = nnz(cluster_2);
% figure;imshow(cluster_2);

cluster_3=imbinarize(reshape(U1(3,:,:),size(originalImage,1),size(originalImage,2)));
Total_White_Pixels3 = nnz(cluster_3);
% figure;imshow(cluster_3);

cluster_4=imbinarize(reshape(U1(4,:,:),size(originalImage,1),size(originalImage,2)));
Total_White_Pixels4 = nnz(cluster_4);
% figure;imshow(cluster_4);

cluster_5=imbinarize(reshape(U1(5,:,:),size(originalImage,1),size(originalImage,2)));
Total_White_Pixels5 = nnz(cluster_5);
% figure;imshow(cluster_4);

if (Total_White_Pixels1<Total_White_Pixels2) & (Total_White_Pixels1<Total_White_Pixels3) & (Total_White_Pixels1<Total_White_Pixels4) & (Total_White_Pixels1<Total_White_Pixels5)
    cluster_image=cluster_1;
elseif  (Total_White_Pixels2<Total_White_Pixels1) & (Total_White_Pixels2<Total_White_Pixels3)& (Total_White_Pixels2<Total_White_Pixels4) & (Total_White_Pixels2<Total_White_Pixels5)
    cluster_image=cluster_2;
elseif  (Total_White_Pixels3<Total_White_Pixels1) & (Total_White_Pixels3<Total_White_Pixels2)& (Total_White_Pixels3<Total_White_Pixels4) & (Total_White_Pixels3<Total_White_Pixels5)
    cluster_image=cluster_3;
elseif  (Total_White_Pixels3<Total_White_Pixels1) & (Total_White_Pixels3<Total_White_Pixels2)& (Total_White_Pixels3<Total_White_Pixels4) & (Total_White_Pixels4<Total_White_Pixels5)
    cluster_image=cluster_4;
else
   cluster_image=cluster_5;
end
figure;imshow(cluster_image);title('Selected Clustered Image');

f=cluster_image;
cluster2=bwareaopen(f,0);
figure;imshow(cluster2);

SE1=strel('disk',1);
cluster3=imerode(cluster2,SE1);
figure;imshow(cluster3);
cluster3=bwareaopen(cluster3,100);
figure;imshow(cluster3);

SE2=strel('disk',1);
cluster4=imdilate(cluster3,SE2);
figure;imshow(cluster4);
cluster4=bwareaopen(cluster4,100);
figure;imshow(cluster4);
%
cell_str4=regionprops(cluster4,'All');
cell_tbl4=struct2table(cell_str4);
p4=prctile(cell_tbl4.FilledArea,[20 100]);
idxLowCounts4 = cell_tbl4.FilledArea >= p4(1);
cell_small4 = cell_tbl4(idxLowCounts4,:);
cell_BB4=cell_small4.BoundingBox;
w4=round(mean(cell_BB4(:,3)));
h4=round(mean(cell_BB4(:,4)));

figure, imshow(Inormalized);title('AutoCryoPicker:Fully Automated Single Particle Picking in Cryo-EM Images'); 
for k = 1 : length(cell_BB4)
  thisBB4 = cell_BB4(k,:);
  rectangle('Position', [thisBB4(1),thisBB4(2),w4,h4],...
  'EdgeColor','w','LineWidth',1 )
end