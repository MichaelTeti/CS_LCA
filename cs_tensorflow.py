import tensorflow as tf
import numpy as np
import sys

n=10000 # number of total elements in signal
signal=np.zeros([n, 1])
k=100 # number of nonzero elements in signal

signal[np.random.randint(0, n, (k, 1)), 0]=np.sign(np.random.randn(k, 1)) # sparse signal

m=n/10 # number of measurements
D=tf.sign(tf.random_normal([m, n])) # random matrix
u=tf.Variable(tf.zeros([n, 1])) # weights
y=tf.matmul(D, tf.cast(signal, tf.float32)) # compressed measurements
lamb=tf.constant(4.0) # threshold
h=tf.constant(.0005) # scale constant

# LCA
with tf.Session() as sess:
  for i in range(200):
    sess.run(tf.global_variables_initializer())
    a=(u-tf.sign(u)*lamb)*(tf.cast(tf.abs(u)>lamb, tf.float32)) 
    u=u+h*(tf.matmul(tf.transpose(D), (y-tf.matmul(D, a)))-u-a) 
  a=tf.round(a)
  A=sess.run(a)
  print np.unique(A)
  e=(n-sum([float(x==y) for (x, y) in zip(A, signal)])) # determine error
  print('%d/%d elements lost'%(e, n))

