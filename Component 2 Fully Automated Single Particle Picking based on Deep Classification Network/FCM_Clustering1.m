function [cluster_image] = FCM_Clustering(img)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
 % specify number of clusters
    clusterNum = 3;
    [ Unow, center, now_obj_fcn ] = FCMforImage( img, clusterNum );
    figure;
    subplot(2,2,1); imshow(img,[]);
    % for i=1:clusterNum
    %     subplot(2,2,i+1);
    %     imshow(Unow(:,:,i),[]);
    % end
    Ckfm1=imbinarize(Unow(:,:,1));
    Ckfm2=imbinarize(Unow(:,:,2));
    Ckfm3=imbinarize(Unow(:,:,3));
%     Ckfm4=imbinarize(Unow(:,:,4));
%     Ckfm5=imbinarize(Unow(:,:,5));


    figure;suptitle('Clustering using FCM');
    subplot 221; imshow(Ckfm1);title('Cluster #1');
    subplot 222; imshow(Ckfm2);title('Cluster #2');
    subplot 223; imshow(Ckfm3);title('Cluster #3');
%     subplot 224; imshow(Ckfm4);title('Cluster #4');
%     subplot 235; imshow(Ckfm5);title('Cluster #5');

    %
    clc;
    disp('_________________________________________________________');
    disp('                                                         ');
    disp('   S  E  G  M  E  N  T  A  T  I  O  N   -  S  T  E  P    ');
    disp('                   U S I N G   F C M                     ');
    disp('_________________________________________________________');
    disp(' ');
    sel=input('Select Propoer Cluster #:" ');
    if sel==1
        cluster=Ckfm1;
    elseif sel==2
        cluster=Ckfm2;
    elseif sel==3
        cluster=Ckfm3;
    else
        cluster=Ckfm4;
    end
%
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

