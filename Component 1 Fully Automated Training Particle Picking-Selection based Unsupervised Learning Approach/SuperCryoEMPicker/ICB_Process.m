function [Morphological_Image] = ICB_Process(originalImage2)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
originalImage1=rgb2gray(originalImage2);
originalImage1 = imcomplement(originalImage1);
originalImage = imresize(originalImage1,.5);
%
%% Pre-processing Part
% Image normalization...
z=mat2gray(originalImage);
% Contrast Enhancement Correction
Inormalized=z;
limit=stretchlim(Inormalized);
CEC_Image_Adjusment=imadjust(Inormalized,[limit(1) limit(2)]);  

% Hostogram Equalization
Cryo_EM_Histogram_Equalization = histeq(CEC_Image_Adjusment);

% Cryo-EM Restoration
Cryo_EM_Restoration = wiener2(Cryo_EM_Histogram_Equalization,[5 5]);

% Adaptive Histogram Equlaizer Cryo-Image
Adaptive_Histogram_Equlaizer = histeq(Cryo_EM_Restoration);
Adaptive_Histogram_Equlaizer=adapthisteq(Adaptive_Histogram_Equlaizer,'clipLimit',.02,'Distribution','rayleigh');
Adaptive_Histogram_Equlaizer=adapthisteq(Adaptive_Histogram_Equlaizer,'clipLimit',.99,'Distribution','rayleigh');

% Gaudided Filtering
Gaudided_Filtering=imguidedfilter(Adaptive_Histogram_Equlaizer);
Gaudided_Filtering=imguidedfilter(Gaudided_Filtering);
Gaudided_Filtering=imguidedfilter(Gaudided_Filtering);
Gaudided_Filtering=imguidedfilter(Gaudided_Filtering);
Gaudided_Filtering=imadjust(Gaudided_Filtering);

% Morphological Image Operation
Morphological_Image=imopen(Gaudided_Filtering,strel('disk',3));
Morphological_Image=imopen(Morphological_Image,strel('disk',1));
Morphological_Image=imopen(Morphological_Image,strel('disk',1));
Morphological_Image=imopen(Morphological_Image,strel('disk',1));
Morphological_Image=imopen(Morphological_Image,strel('disk',1));
end

