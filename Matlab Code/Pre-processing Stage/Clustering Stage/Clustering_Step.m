clear all;clc;close all;
load('Cropped_image1.mat')

img=im2;
[row,col]= size(img);

%%  Kmens Norm1
% specify number of clusters
k=4;
% Range
img_vector=img(:);
range = max(img_vector) - min(img_vector);
    
% Determine the # of steps
stepv = range/k;
% Cluster initialization
centers=[stepv:stepv:max(img_vector)]';


    % Cluster.
    clustered = reshape(kmeans(img(:), k,'start',centers,'Distance','cityblock','MaxIter',100), size(img));
    % Sort clusters. 
    clusterintensity = zeros(k, 1);
    for j = 1:k
        clusterintensity(j) = img(find(clustered == j, 1));
    end 
    clusteridx = zeros(k, 1);
    for j = 1:k
        clusteridx(clusterintensity == min(clusterintensity)) = j;    
        clusterintensity(clusterintensity == min(clusterintensity)) = NaN;
    end
    clustered = clusteridx(clustered);
    
Ck1=clustered;Ck1(clustered~=1)=0;
Ck2=clustered;Ck2(clustered~=2)=0;
Ck3=clustered;Ck3(clustered~=3)=0;
Ck4=clustered;Ck4(clustered~=4)=0;

figure;
subplot 221; imshow(Ck1);title('Cluster #1');
subplot 222; imshow(Ck2);title('Cluster #2');
subplot 223; imshow(Ck3);title('Cluster #3');
subplot 224; imshow(Ck4);title('Cluster #4');
suptitle('Clustering using K-means');

SE=strel('disk',4);
k=imfill(Ck4,'holes');BW = imopen(k,SE);imshow(BW,[])
c=bwareaopen(BW,100);
figure;imshow(c);title('Clustered Image');
figure;histogram(c);title('K-Means Histogram');

figure;imshow(im2);title('Detection using K-Means Histogram');
cell_str=regionprops(c,'All');

for k = 1 : length(cell_str)
  thisBB = cell_str(k).BoundingBox;
  rectangle('Position', [thisBB(1),thisBB(2),thisBB(3),thisBB(4)],...
  'EdgeColor','r','LineWidth',2 )
end



%% FCM
clusterNum=k;
[row,col]= size(img);
[ Unow, center, now_obj_fcn ] = FCM_Image( img, clusterNum );

c1=Unow(:,:,1);
c2=Unow(:,:,2);             
c3=Unow(:,:,3);
c4=Unow(:,:,4);

CC=[c1(:) c2(:) c3(:) c4(:)];
Img_Cltr=[];
for iii=1:size(CC,1)
[~,I]=max(CC(iii,:));
Img_Cltr(iii)=I;
end
FCMclustered=reshape(Img_Cltr',[row col]);
    % Sort clusters. 
    clusterintensity = zeros(k, 1);
    for j = 1:k
        clusterintensity(j) = img(find(FCMclustered == j, 1));
    end 
    clusteridx = zeros(k, 1);
    for j = 1:k
        clusteridx(clusterintensity == min(clusterintensity)) = j;    
        clusterintensity(clusterintensity == min(clusterintensity)) = NaN;
    end
    FCMclustered = clusteridx(FCMclustered);
    
    
Ckm1=FCMclustered;Ckm1(FCMclustered~=1)=0;
Ckm2=FCMclustered;Ckm2(FCMclustered~=2)=0;
Ckm3=FCMclustered;Ckm3(FCMclustered~=3)=0;
Ckm4=FCMclustered;Ckm4(FCMclustered~=4)=0;

figure;suptitle('Clustering using FCM');
subplot 221; imshow(Ckm1);title('Cluster #1');
subplot 222; imshow(Ckm2);title('Cluster #2');
subplot 223; imshow(Ckm3);title('Cluster #3');
subplot 224; imshow(Ckm4);title('Cluster #4');
suptitle('Clustering using FCM');

SE=strel('disk',4);
k=imfill(Ckm4,'holes');BW = imopen(k,SE);imshow(BW,[])
c=bwareaopen(BW,100);
figure;imshow(c);title('Clustered Image');
figure;histogram(c);title('FCM Histogram');

figure;imshow(im2);title('Detection using FCM Histogram');
cell_str=regionprops(c,'All');

for k = 1 : length(cell_str)
  thisBB = cell_str(k).BoundingBox;
  rectangle('Position', [thisBB(1),thisBB(2),thisBB(3),thisBB(4)],...
  'EdgeColor','r','LineWidth',2 )
end

%% FLICM

%%
figure;
% subplot(2,2,1);imshow(C4,[]);title('Adil Algorithm');
subplot(2,2,1);imshow(Ck4);title('K-Means');
subplot(2,2,2);histogram(Ck4);title('K-Means Histogram');
subplot(2,2,3);imshow(Ckm4);title('FCM');
subplot(2,2,4);histogram(Ckm4);title('FCM Histogram');



