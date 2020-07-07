function [ clustered, U ] = FCM_Sorted( img, Clusters, p )
%  FCM Norm2

if nargin ==2
    p=0;
end
    
% douple the image
img=im2double(img);
% Convert the image from 2D to 1D image space
img_vector = img(:);

[~,U] = fcm(img_vector,Clusters);
[~,I] = max(U,[],1);

clustered=reshape(I,size(img));

% Sort clusters. 
clusterintensity = zeros(Clusters, 1);
for j = 1:Clusters
    clusterintensity(j) = img(find(clustered == j, 1));
end 
clusteridx = zeros(Clusters, 1);
for j = 1:Clusters
    clusteridx(clusterintensity == min(clusterintensity)) = j;    
    clusterintensity(clusterintensity == min(clusterintensity)) = NaN;
end
clustered = clusteridx(clustered);

if p==1
figure;
if Clusters<=2
    s=1;
elseif Clusters<=5
    s=2;
else
    s=3;
end
subplot(s,3,1);imshow(img);title('Thermal img');
for f=1:Clusters
   subplot(s,3,f+1);imshow(clustered==f);title(['Image in Cluster # ' num2str(f)]); 
end  
end

end