function [cluster_image] = Our_Clustering1(im,opt)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% douple the image
img=double(im);
% im = imcomplement(im);
% figure;imshow(im);

[row,col,color_mp]= size(img);

% Convert the image from 2D to 1D image space...
img_vector = img(:);

% specify number of clusters
Clusters=5;    
  
Cluster = cell(1,Clusters);
Cluster(:) = {zeros(size(img_vector,1),1);};
    
% Range       
range = max(img_vector) - min(img_vector);
    
% Determine the # of steps
stepv = range/Clusters;
% Cluster initialization
K=stepv:stepv:max(img_vector);

for ii=1:size(img_vector,1)
    difference = abs(K-img_vector(ii));
    [y,ind]=min(difference);
    Cluster{ind}(ii)=img_vector(ii);
end

cluster_1=reshape(Cluster{1,1},[row col]);
C1=cluster_1;C1(cluster_1~=0)=1;
cluster_2=reshape(Cluster{1,2},[row col]);
C2=cluster_2;C2(cluster_2~=0)=2;
cluster_3=reshape(Cluster{1,3},[row col]);
C3=cluster_3;C3(cluster_3~=0)=3;
cluster_4=reshape(Cluster{1,4},[row col]);
C4=cluster_4;C4(cluster_4~=0)=4;
cluster_5=reshape(Cluster{1,5},[row col]);
C4=cluster_5;C5(cluster_5~=0)=5;
%
figure;suptitle('Clustering using Our Clustering');
subplot 221; imshow(C1);title('Cluster #1');
subplot 222; imshow(C2);title('Cluster #2');
subplot 223; imshow(C3);title('Cluster #3');
subplot 224; imshow(C4);title('Cluster #4');
%%
if opt==1
    figure; imshow(C1);title('Cryo-EM Binary Mask Image');
    imwrite(C1,'Our_clustering_before_cleaning.png');
    cluster1=bwareaopen(C1,500);
    binIM=cluster1;
    SE=strel('disk',1);
    cluster2=imerode(binIM,SE);
    k=imfill(cluster2,'holes');
    BW = imclose(k,SE);
    cluster3=bwareaopen(BW,500);
    cluster3=imdilate(cluster3,SE);
    figure; imshow(cluster3);title('Cleaned Cryo-EM Binary Mask Image');
    cluster3=imfill(cluster3,'holes');
    cluster_image=cluster3; 
else
    figure; imshow(C4);title('Cryo-EM Binary Mask Image');
    imwrite(C4,'Our_clustering_before_cleaning.png');
    % cluster1=bwareaopen(C1,500);
    cluster1=bwareaopen(C4,10);
    binIM=cluster1;
    SE=strel('disk',1);
    cluster2=imerode(binIM,SE);
    k=imfill(cluster2,'holes');
    BW = imclose(k,SE);
    % cluster3=bwareaopen(BW,1500);
    cluster3=bwareaopen(BW,10);
    cluster3=imdilate(cluster3,SE);
    figure; imshow(cluster3);title('Cleaned Cryo-EM Binary Mask Image');
    cluster3=imfill(cluster3,'holes');
    cluster_image=cluster3; 
end
end

