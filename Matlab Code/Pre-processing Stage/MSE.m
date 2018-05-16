function [ MSE ] = MSE( orginal_image,new_image )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
     MSE = mean((double(orginal_image(:))-double(new_image(:))).^2);
end

