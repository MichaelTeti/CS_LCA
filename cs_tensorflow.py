import tensorflow as tf
import numpy as np
import sys

n=1000
signal=np.zeros([n, 1])
k=10

signal[np.random.randint(0, n, (k, 1)), 0]=np.sign(np.random.randn(k, 1)) 

m=100 
D=tf.Variable(tf.sign(tf.random_normal([m, n])))
u=tf.Variable(tf.zeros([n, 1]))
y=tf.matmul(D, tf.cast(signal, tf.float32))
lamb=tf.constant(4.0)
h=tf.constant(.005)


with tf.Session() as sess:
  for i in range(100):
    sess.run(tf.global_variables_initializer())
    a=(u-tf.sign(u)*lamb)*(tf.cast(tf.abs(u)>lamb, tf.float32))
    u=u+h*(tf.matmul(tf.transpose(D), (y-tf.matmul(D, a)))-u-a)
  a=tf.round(a)
  A=sess.run(a)
  e=(n-sum([int(x==y) for (x, y) in zip(A, signal)]))/n
  print('%d percent lost'%(e*100))


