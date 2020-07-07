%% read the mrc image
% chang the filename1 to read the targeted image 
filename1='May08_03.05.02.bin.mrc';
fid1= fopen(filename1,'r','n');
nx1=fread(fid1,1,'long');
ny1=fread(fid1,1,'long');
nz1=fread(fid1,1,'long');
mode1=fread(fid1,1,'long');
if mode1==1
    A1 = fread(fid1,nx1*ny1*nz1,'int16');
elseif mode1==2
    A1 = fread(fid1,nx1*ny1*nz1,'float32');
end
fclose(fid1);
image= reshape(A1, [nx1 ny1 nz1]);
%% averaging the image
image_sum=sum(image,3); % the image is summed along the z dimension
image_avg=image_sum/nz1; % if the image dimension z=1 then the image will not be affected

%% further processing

image_gray=mat2gray(image_avg);% image is converted to grayscale