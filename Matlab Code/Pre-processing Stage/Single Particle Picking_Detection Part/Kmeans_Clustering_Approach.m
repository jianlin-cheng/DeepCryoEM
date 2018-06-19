function [Ckm4] = Kmeans_Clustering_Approach(img)
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
    sel=input('Select the cluster ');
    if sel==1
        cluster=Ckm1;
    elseif sel==2
        cluster=Ckm2;
    elseif sel==3
        cluster=Ckm3;
    else
        cluster=Ckm4;
    end
end

