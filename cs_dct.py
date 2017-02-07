import tensorflow as tf
import numpy as np
from scipy.misc import *
from scipy.fftpack import *
import sys
import cv2

im=imread('monalisa.jpg') # read the image

im=imresize(im, [im.shape[0]/4, im.shape[1]/4]) # make a little smaller

imshp=im.shape # get the new shape

im=np.reshape(np.mean(im, 2), [im.size/3, 1])+1e-5 # vectorize, grayscale

im=(im-np.mean(im))/np.std(im) # scale elements between 0 and 1

n=len(im) # num elements

m=n/10 # num measurements

lamb=2 # threshold

t=1
 
h=1e-7 # scale factor
 
d=h/t
 
u=tf.zeros([n, 1], dtype=tf.float32)

phi=tf.random_normal([m, n], dtype=tf.float32) # random matrix

phi=phi/tf.reduce_max(phi) # scale random values 

psi=np.float32(idct(np.identity(n))) # dct dictionary

D=tf.matmul(phi, psi)

y=tf.matmul(phi, np.float32(im)) # compressed measurements



with tf.Session() as sess:
  for i in range(100): 
    a=(u-tf.sign(u)*lamb)*(tf.cast(tf.abs(u)>lamb, tf.float32)) 
    u=u+h*(tf.matmul(tf.transpose(D), (y-tf.matmul(D, a)))-u-a) # ODE
  rec=tf.matmul(psi, a)
  A=sess.run(rec)
  A=np.reshape(A, [imshp[0], imshp[1]])
  diff=np.concatenate((np.reshape(im, [imshp[0], imshp[1]]), A), axis=1)
  cv2.namedWindow('reconstruction', cv2.WINDOW_NORMAL)
  cv2.imshow('reconstruction', diff)
  cv2.waitKey(10000)

