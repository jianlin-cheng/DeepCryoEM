function [cluster] = FCM_Clustering(img)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
 % specify number of clusters
    clusterNum = 4;
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
    Ckfm4=imbinarize(Unow(:,:,4));

    figure;suptitle('Clustering using FCM');
    subplot 221; imshow(Ckfm1);title('Cluster #1');
    subplot 222; imshow(Ckfm2);title('Cluster #2');
    subplot 223; imshow(Ckfm3);title('Cluster #3');
    subplot 224; imshow(Ckfm4);title('Cluster #4');

    sel1=input('Select the cluster ');
    if sel1==1
        cluster=Ckfm1;
    elseif sel1==2
        cluster=Ckfm2;
    elseif sel1==3
        cluster=Ckfm3;
    else
        cluster=Ckfm4;
    end
end

