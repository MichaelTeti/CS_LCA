% compressive sensing of natural images

clear;
close all;
clc;

im=im2double(imread('monalisa.jpg'));
im=imresize(im, [size(im, 1)/2, size(im, 2)/2], 'bilinear', 0);
im=rgb2gray(im);
imsz=size(im)
im=im(:); 
n=length(im);
m=n/2;
phi=randn(m, n);
y=phi*im;
psi=idct(eye(n));
D=phi*psi;
lambda=2;
t=1;
h=0.0001;
d=h/t;
u=zeros(n, 1);

while e>.1
  a=(u-sign(u).*(lambda)).*(abs(u)>(lambda));
  u=u+d*(D'*(y-D*a)-u-a);
  rec=psi*a
end

rec=reshape(rec, imsz(1), imsz(2));
save('rec.mat', 'rec');
