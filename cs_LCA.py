import tensorflow as tf
import numpy as np
import cv2

# load sparse image with mostly zeros (black) and a few nonzeros (white)
im=cv2.cvtColor(cv2.imread('circle.png'), cv2.COLOR_BGR2GRAY)
im=im[::20, ::20]
cv2.imshow('original image', im)
cv2.waitKey(5000)
imshp=np.shape(im)
im=np.reshape(im, [np.size(im), 1]) # vectorize image

# scale image features between 0 and 1
im=(im-np.tile(np.amin(im), (np.size(im), 1)))/np.tile(np.amax(im)-np.amin(im), (np.size(im), 1))
k=np.size(np.where(im!=0))/2 # number of nonzero elements in image
print('%d/%d nonzero elements in signal'%(k, np.size(im)))
n=np.size(im) # total num. of elements in im
m=n/10 # num. of samples
D=tf.sign(tf.random_normal([m, n])) # random +1, -1 matrix
u=tf.Variable(tf.zeros([n, 1])) # weights
y=tf.matmul(D, tf.cast(im, tf.float32)) # sampling image
lamb=tf.constant(4.0) # threshold
h=tf.constant(.005)

with tf.Session() as sess:
  for i in range(200): 
    sess.run(tf.global_variables_initializer()) # initialize variables
    a=(u-tf.sign(u)*lamb)*(tf.cast(tf.abs(u)>lamb, tf.float32)) # LCA
    u=u+h*(tf.matmul(tf.transpose(D), (y-tf.matmul(D, a)))-u-a)
  a=tf.round(a)
  A=sess.run(a)
  print np.unique(A)
  print('%d/%d elements lost'%(n-sum([float(x==y) for (x, y) in zip(A, im)]), np.size(im)))

cv2.imshow('LCA reconstruction', np.reshape(A, [imshp[0], imshp[1]]))
cv2.waitKey(10000)
