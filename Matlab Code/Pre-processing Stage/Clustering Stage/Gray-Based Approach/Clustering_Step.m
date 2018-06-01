clear all;clc;close all;
load('Cropped_image.mat')

img=J1;
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

figure;suptitle('Clustering using K-means');
subplot 221; imshow(Ck1);title('Cluster #1');
subplot 222; imshow(Ck2);title('Cluster #2');
subplot 223; imshow(Ck3);title('Cluster #3');
subplot 224; imshow(Ck4);title('Cluster #4');

SE=strel('disk',5);
k=imfill(Ck4,'holes');BW = imopen(k,SE);imshow(BW,[])
c=bwareaopen(BW,100);
figure;imshow(c);title('Clustered Image');
figure;histogram(c);title('K-Means Histogram');

figure;imshow(img);title('Detection using K-Means Histogram');
cell_str=regionprops(c,'All');

for k = 1 : length(cell_str)
  thisBB = cell_str(k).BoundingBox;
  rectangle('Position', [thisBB(1),thisBB(2),thisBB(3),thisBB(4)],...
  'EdgeColor','r','LineWidth',2 )
end

%% FCM
clusterNum = 4;
[ Unow, center, now_obj_fcn ] = FCMforImage( img, clusterNum );
figure;
subplot(2,2,1); imshow(img,[]);
% for i=1:clusterNum
%     subplot(2,2,i+1);
%     imshow(Unow(:,:,i),[]);
% end
Ckm1=imbinarize(Unow(:,:,1));
Ckm2=imbinarize(Unow(:,:,2));
Ckm3=imbinarize(Unow(:,:,3));
Ckm4=imbinarize(Unow(:,:,4));

figure;suptitle('Clustering using FCM');
subplot 221; imshow(Ckm1);title('Cluster #1');
subplot 222; imshow(Ckm2);title('Cluster #2');
subplot 223; imshow(Ckm3);title('Cluster #3');
subplot 224; imshow(Ckm4);title('Cluster #4');

sel=input('Select the cluster ');
if sel==1
    Ckm4=Ckm1;
elseif sel==2
    Ckm4=Ckm2;
elseif sel==3
    Ckm4=Ckm3;
else
    Ckm4=Ckm4;
end
SE=strel('disk',5);
k=imfill(Ckm4,'holes');BW = imopen(k,SE);imshow(BW,[])
c=bwareaopen(BW,100);
figure;imshow(c);title('Clustered Image');
figure;histogram(c);title('FCM Histogram');

figure;imshow(img);title('Detection using FCM Histogram');
cell_str=regionprops(c,'All');

for k = 1 : length(cell_str)
  thisBB = cell_str(k).BoundingBox;
  rectangle('Position', [thisBB(1),thisBB(2),thisBB(3),thisBB(4)],...
  'EdgeColor','r','LineWidth',2 )
end

%% FLICM
%%
%% Using Our Apporch
img=double(img);
figure; imshow(img);

% Convert the image from 2D to 1D image space...
img_vector =img(:);

% specify number of clusters
Clusters=4;    
  
Cluster = cell(1,Clusters);
Cluster(:) = {zeros(size(img_vector,1),1);};
    
% Range       
range = max(img_vector) - min(img_vector);
    
% Determine the # of steps
stepv = range/Clusters;
% Cluster initialization
K=stepv:stepv:max(img_vector);

for i=1:size(img_vector,1)
    difference = abs(K-img_vector(i));
    [y,ind]=min(difference);
    Cluster{ind}(i)=img_vector(i);
end

cluster_1=reshape(Cluster{1,1},[row col]);
C1=cluster_1;C1(cluster_1~=0)=1;
cluster_2=reshape(Cluster{1,2},[row col]);
C2=cluster_2;C2(cluster_2~=0)=2;
cluster_3=reshape(Cluster{1,3},[row col]);
C3=cluster_3;C3(cluster_3~=0)=3;
cluster_4=reshape(Cluster{1,4},[row col]);
C4=cluster_4;C4(cluster_4~=0)=4;

figure;suptitle('Clustering using Our Algorithm');
subplot 221; imshow(C1);title('Cluster #1');
subplot 222; imshow(C2);title('Cluster #2');
subplot 223; imshow(C3);title('Cluster #3');
subplot 224; imshow(C4);title('Cluster #4');

SE=strel('disk',5);
k=imfill(Ck4,'holes');BW = imopen(k,SE);imshow(BW,[])
c=bwareaopen(BW,100);
figure;imshow(c);title('Clustered Image');
figure;histogram(c);title('FCM Histogram');

figure;imshow(img);title('Detection using Our Algorthim');
cell_str=regionprops(c,'All');

for k = 1 : length(cell_str)
  thisBB = cell_str(k).BoundingBox;
  rectangle('Position', [thisBB(1),thisBB(2),thisBB(3),thisBB(4)],...
  'EdgeColor','r','LineWidth',2 )
end


%%%
figure;
subplot(3,2,1);imshow(Ck4);title('K-Means');
subplot(3,2,2);histogram(Ck4);title('K-Means Histogram');
subplot(3,2,3);imshow(Ckm4);title('FCM');
subplot(3,2,4);histogram(Ckm4);title('FCM Histogram');
subplot(3,2,5);imshow(C4);title('Our Algorithm');
subplot(3,2,6);histogram(C4);title('Our Algorithm Histogram');


