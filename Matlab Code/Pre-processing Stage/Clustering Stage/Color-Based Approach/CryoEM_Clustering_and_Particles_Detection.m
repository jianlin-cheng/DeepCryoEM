%% CryoEM Color-Based Segmentation Using SOM...
%
close all; clear all; clc;
%% Step(1): Iron Color Palettes...
    % The built-in colormap of the matlab seems not that kind of good for
    % some images, for this reason using a text editor and its value is 
    % defined inside. The value of this one is defined in uint8 format 
    % (0-255), thus the image converted to floating point (0-1) and imported 
    % them to matlab. Now it works pretty well.
    load('iron.mat');
    C=iron;
    load('Cropped_image.mat');
    img=J1;
    [row,col]= size(img);
    img=double(img);
    figure; imshow(img);
    G=double(img);
    L = size(C,1);
    %Scale the matrix to the range of the map.
    Gs = round(interp1(linspace(min(G(:)),max(G(:)),L),1:L,G));
    % Make RGB image from scaled.
    H = reshape(C(Gs,:),[size(Gs) 3]); 
    uint8Image = uint8(255 * mat2gray(H));
    imshowpair(G,uint8Image,'montage');
    % Image=uint8Image(:,:,2);
    figure;imshow(uint8Image);
    % imwrite(uint8Image,'3.jpg');
    
%% Step (2): CryoEM Clustering using SOM...
    % read the MRI image from the spesific directory
    Img=uint8Image;
    
    % Display the Color CryoEM channels
    figure;imshow(Img(:,:,1));title('Red Color Space');
    figure;imshow(Img(:,:,2));title('Green Color Space');
    figure;imshow(Img(:,:,3));title('Blue Color Space');
    
    % Convert the image from RGB space to LAB space
    I=Img;
    imshow(I);
    cform = makecform('srgb2lab');
    lab_I = applycform(I,cform);
    ab = double(lab_I(:,:,2:3));
    figure;imshow(lab_I(:,:,1));title('L* Color Space');
    figure;imshow(lab_I(:,:,1));title('L* Color Space');
    figure;imshow(lab_I(:,:,2));title('a* Color Space');
    figure;imshow(lab_I(:,:,3));title('b* Color Space');
    
    % Cluster the image using SOM
    nrows = size(ab,1);
    ncols = size(ab,2);
    ab = reshape(ab,nrows*ncols,2);
    a = ab(:,1);
    b = ab(:,2);
    normA = (a-min(a(:))) ./ (max(a(:))-min(a(:)));
    normB = (b-min(b(:))) ./ (max(b(:))-min(b(:)));
    ab = [normA normB];
    newnRows = size(ab,1);
    newnCols = size(ab,2);
    cluster = 5;
    % Max number of iteration
    N = 90;
    % initial learning rate
    eta = 0.3;
    % exponential decay rate of the learning rate
    etadecay = 0.2;
    %random weight

    w = rand(2,cluster);
    %initial D
    D = zeros(1,cluster);
    % initial cluster index
    clusterindex = zeros(newnRows,1);
    % start 
    for t = 1:N
       for data = 1 : newnRows
           for c = 1 : cluster
               D(c) = sqrt(((w(1,c)-ab(data,1))^2) + ((w(2,c)-ab(data,2))^2));
           end
           %find best macthing unit
           [~, bmuindex] = min(D);
           clusterindex(data)=bmuindex;

           %update weight
           oldW = w(:,bmuindex);
           new = oldW +  eta * (reshape(ab(data,:),2,1)-oldW);
    %        new = oldW +  eta * (reshape(ab(data,:),2,1)-oldW);
           w(:,bmuindex) = new;

       end
       % update learning rate
       eta= etadecay * eta;
    end

    %Label Every Pixel in the Image Using the Results from KMEANS
    pixel_labels = reshape(clusterindex,nrows,ncols);
    %Create Images that Segment the I Image by Color.
    segmented_images = cell(1,3);
    rgb_label = repmat(pixel_labels,[1 1 3]);

    for k = 1:cluster
        color = I;
        color(rgb_label ~= k) = 0;
        segmented_images{k} = color;
    end
    figure,imshow(segmented_images{1}), title('objects in cluster 1');
    figure,imshow(segmented_images{2}), title('objects in cluster 2');
    figure,imshow(segmented_images{3}), title('objects in cluster 3');
    figure,imshow(segmented_images{4}), title('objects in cluster 4');
    figure,imshow(segmented_images{5}), title('objects in cluster 5');
    % figure,imshow(segmented_images{6}), title('objects in cluster 6');
    %%
    BW = double(imbinarize(rgb2gray(segmented_images{1})));
    figure;imshow(BW);
    SE=strel('disk',4);
    k=imfill(BW,'holes');BW = imopen(k,SE);imshow(BW,[])
    c=bwareaopen(BW,100);
    figure;imshow(c);title('Clustered Image');
    figure;histogram(c);title('SOM Clustering Histogram');

    % %% Step 5: Display 'a*' and 'b*' Values of the Labeled Colors.
    % % You can see how well the nearest neighbor classification separated the different color populations 
    % % by plotting the 'a*' and 'b*' values of pixels that were classified into separate colors. 
    % % For display purposes, label each point with its color label.
    % purple = [119/255 73/255 152/255];
    % plot_labels = {'k', 'r', 'g', purple, 'm', 'y'};
    % 
    % figure
    % nColors=cluster;
    % for count = 1:nColors
    %   plot(a(label==count-1),b(label==count-1),'.','MarkerEdgeColor', ...
    %        plot_labels{count}, 'MarkerFaceColor', plot_labels{count});
    %   hold on;
    % end
    %   
    % title('Scatterplot of the segmented pixels in ''a*b*'' space');
    % xlabel('''a*'' values');
    % ylabel('''b*'' values');
    %%
    [img,path]=uigetfile('*.jpg','Select a MRI Brain Tumor Image');
    % open the directory box
    str=strcat(path,img);

    % read the MRI image from the spesific directory
    img=imread(str);
    figure;imshow(img);title('Detection using SOM-Color Based Clustering');
    cell_str=regionprops(c,'All');

    for k = 1 : length(cell_str)
      thisBB = cell_str(k).BoundingBox;
      rectangle('Position', [thisBB(1),thisBB(2),thisBB(3),thisBB(4)],...
      'EdgeColor','r','LineWidth',2 )
    end
