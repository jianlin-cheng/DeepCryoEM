function [cluster_image] = K_means_Clustering(img)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
 % specify number of clusters
    ab = double(img);
    nrows = size(ab,1);
    ncols = size(ab,2);
    ab = reshape(ab,nrows*ncols,1);

    nColors = 4;
    % repeat the clustering 3 times to avoid local minima
    [cluster_idx, cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean', 'Replicates', 3);

    pixel_labels = reshape(cluster_idx,nrows,ncols);
    imshow(pixel_labels,[]);
    
    % Sort the Clusters
    Ckm1=pixel_labels;Ckm1(pixel_labels~=1)=0;
    Ckm2=pixel_labels;Ckm2(pixel_labels~=2)=0;
    Ckm3=pixel_labels;Ckm3(pixel_labels~=3)=0;
    Ckm4=pixel_labels;Ckm4(pixel_labels~=4)=0;

    figure;suptitle('Clustering using K-means');
    subplot 221; imshow(Ckm1);title('Cluster #1');
    subplot 222; imshow(Ckm2);title('Cluster #2');
    subplot 223; imshow(Ckm3);title('Cluster #3');
    subplot 224; imshow(Ckm4);title('Cluster #4');
    disp(' ');
    clc;
    disp('_________________________________________________________');
    disp('                                                         ');
    disp('   S  E  G  M  E  N  T  A  T  I  O  N   -  S  T  E  P    ');
    disp('              U S I N G   K - M E A N S                  ');
    disp('_________________________________________________________');
    disp(' ');
    sel=input('Select Propoer Cluster #:" ');
    %
    if sel==1
        cluster=Ckm1;
    elseif sel==2
        cluster=Ckm2;
    elseif sel==3
        cluster=Ckm3;
    else
        cluster=Ckm4;
    end
% %% for Image 35
cluster1=bwareaopen(cluster,0);
figure; imshow(cluster1);
cluster1=bwareaopen(cluster,100);
figure; imshow(cluster1);title('CryoEM-Binary Mask Image');
binIM=cluster1;
SE=strel('disk',1);
cluster2=imerode(binIM,SE);
k=imfill(cluster2,'holes');
BW = imclose(k,SE);imshow(BW,[]);
cluster3=bwareaopen(BW,0);
cluster3=imdilate(cluster3,SE);
figure; imshow(cluster3);title('CryoEM-Binary Mask Image');
cluster_image=cluster3; 
end

