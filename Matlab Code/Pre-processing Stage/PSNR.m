function [psnr_value]=PSNR(original_image,new_image)
%% The function Calculate the PSNR and The MSE
 % The Mean Square Error (MSE) and the Peak Signal to Noise Ratio (PSNR) 
 % are the two error metrics used to compare image compression quality.
 % The MSE represents the cumulative squared error between the compressed 
 % and the original image, whereas PSNR represents a measure of the peak error. 
 % The lower the value of MSE, the lower the error.
 %
 % Input : 
 %       original_image -> is the Original Image
 %       new_image -> Enhanced Image
 % Output: 
 %       PSNR -> Peak Signal to Noise Ratio (PSNR)
 %       MSE  -> Mean Square Error (MSE)
%
%% Step (1): Conmvert The Original and Enhanced Image to douple...
original_image=double(original_image);
new_image=double(new_image);

%% Step(2): Check the Both Images is they are the same or not...
if (original_image==new_image)
    psnr_value=100;
else
    [r,c,p]=size(new_image);

%% Step(3): Calculate the MSE and the PSNR...    
d=0;
for i=1:r
    for j=1:c
        % find the diference between both images 
        d=d+(new_image(i,j)-original_image(i,j))^2;
    end
end
% calculate the MSE 
% Sum the Squared Image and divide by the number of elements
% to get the Mean Squared Error.  It will be a scalar (a single number).
mse=d/(r*c);
% Calculate PSNR (Peak Signal to Noise Ratio) from the MSE according 
% to the formula.
maximum=max(new_image(:));
% calculate the PSR value
psnr_value=abs(10*log10(maximum^2/mse));
end