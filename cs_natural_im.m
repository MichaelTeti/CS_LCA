% compressive sensing of natural images

clear;
close all;
clc;

im=im2double(imread('monalisa.jpg'));
im=imresize(im, [size(im, 1)/5, size(im, 2)/5], 'bilinear', 0);
im=rgb2gray(im);
im=im(:); 
n=length(im);
m=n/10;
phi=randn(m, n);
y=phi*im;
psi=idct(eye(n));
D=phi*psi;

