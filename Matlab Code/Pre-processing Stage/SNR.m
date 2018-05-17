function snr_value= SNR(original_image,new_image)
%% SNR (Signal to noise ratio)
% The signal-to-noise ratio (SNR) is used in imaging as a physical measure 
% of the sensitivity of a (digital or film) imaging system.
%
%% Step (1): Get the Image Sizes (Original and Enhancement one)
[signalRowSize,signalColSize] = size(original_image);
[noiseRowSize, noiseColSize] = size(new_image);

signalAmp = original_image(:);
noiseAmp = new_image(:);

%% Step(2): Calcu;ate the Signal and Noise Power
signalPower = sum(signalAmp.^2)/(signalRowSize*signalColSize);
noisePower = sum(noiseAmp.^2)/(noiseRowSize*noiseColSize);

%% Step(3): Calculate the SNR Value
snr_value = 10*log10(signalPower/noisePower);
end 