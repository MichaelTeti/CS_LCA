import tensorflow as tf
import numpy as np
from scipy.misc import *
from scipy.fftpack import *

im=imread('monalisa.jpg')
im=imresize(im, [im.shape[0]/2, im.shape[1]/2])
imshp=im.shape

im=np.reshape(np.mean(im, 2), [im.size/3, 1])

n=len(im) # num samples
m=n/10 # num measurements
lamb=2
t=1
h=1e-4
d=h/t
u=tf.zeros([n, 1], dtype=tf.float32)
phi=tf.random_normal([m, n], dtype=tf.float64) # random matrix

y=tf.matmul(phi, im) # compressed measurements

psi=idct(np.identity(n))

D=tf.matmul(phi, psi)

with tf.Session() as sess:
  for i in range(100): 
    a=(u-tf.sign(u)*lamb)*(tf.cast(tf.abs(u)>lamb, tf.float32)) 
    u=u+h*(tf.matmul(tf.transpose(D), (y-tf.matmul(D, a)))-u-a) # ODE

rec=tf.matmul(psi, a)
imshow(np.reshape(rec, [imshp[0], imshp[1]]))


