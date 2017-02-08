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
lambda=2;
t=1;
h=0.0001;
d=h/t;
u=zeros(n, 1);

for i=1:100
  a=(u-sign(u).*(lambda)).*(abs(u)>(lambda));
  u=u+d*(D'*(y-D*a)-u-a);
end

rec=psi*a;
save('rec.mat', rec);
