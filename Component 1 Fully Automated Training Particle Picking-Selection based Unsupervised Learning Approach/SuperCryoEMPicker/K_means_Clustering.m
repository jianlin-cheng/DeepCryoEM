function [cluster_image] = K_means_Clustering(img)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
 % specify number of clusters
    ab = double(img);
    nrows = size(ab,1);
    ncols = size(ab,2);
    ab = reshape(ab,nrows*ncols,1);

    nColors = 5;
    % repeat the clustering 3 times to avoid local minima
    [cluster_idx, cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean', 'Replicates', 3);

    pixel_labels = reshape(cluster_idx,nrows,ncols);
    imshow(pixel_labels,[]);
    
    % Sort the Clusters
    Ckm1=pixel_labels;Ckm1(pixel_labels~=1)=0;
    Ckm2=pixel_labels;Ckm2(pixel_labels~=2)=0;
    Ckm3=pixel_labels;Ckm3(pixel_labels~=3)=0;
    Ckm4=pixel_labels;Ckm4(pixel_labels~=4)=0;
    Ckm5=pixel_labels;Ckm5(pixel_labels~=5)=0;

    figure;suptitle('Clustering using K-means');
    subplot 231; imshow(Ckm1);title('Cluster #1');
    subplot 232; imshow(Ckm2);title('Cluster #2');
    subplot 233; imshow(Ckm3);title('Cluster #3');
    subplot 234; imshow(Ckm4);title('Cluster #4');
    subplot 235; imshow(Ckm5);title('Cluster #5');


    Total_White_Pixels1 = nnz(Ckm1);
%     figure;imshow(Ckm1);

    Total_White_Pixels2 = nnz(Ckm2);
%     figure;imshow(Ckm2);

    Total_White_Pixels3 = nnz(Ckm3);
%     figure;imshow(Ckm3);

    Total_White_Pixels4 = nnz(Ckm4);
%     figure;imshow(Ckm4);

    Total_White_Pixels5 = nnz(Ckm5);


    if (Total_White_Pixels1<Total_White_Pixels2) & (Total_White_Pixels1<Total_White_Pixels3) & (Total_White_Pixels1<Total_White_Pixels4) & (Total_White_Pixels1<Total_White_Pixels5)
        cluster_image=Ckm1;
    elseif  (Total_White_Pixels2<Total_White_Pixels1) & (Total_White_Pixels2<Total_White_Pixels3)& (Total_White_Pixels2<Total_White_Pixels4) & (Total_White_Pixels2<Total_White_Pixels5)
        cluster_image=Ckm2;
    elseif  (Total_White_Pixels3<Total_White_Pixels1) & (Total_White_Pixels3<Total_White_Pixels2)& (Total_White_Pixels3<Total_White_Pixels4) & (Total_White_Pixels3<Total_White_Pixels5)
        cluster_image=Ckm3;
    elseif  (Total_White_Pixels3<Total_White_Pixels1) & (Total_White_Pixels3<Total_White_Pixels2)& (Total_White_Pixels3<Total_White_Pixels4) & (Total_White_Pixels4<Total_White_Pixels5)
        cluster_image=Ckm4;
    else
       cluster_image=Ckm5;
    end
    figure;imshow(cluster_image);title('Selected Clustered Image');
end

