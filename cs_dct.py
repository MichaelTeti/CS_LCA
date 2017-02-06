import tensorflow as tf
import numpy as np
import cv2
from scipy.misc import *
import matplotlib.pyplot as plt

im=np.float32(cv2.cvtColor(cv2.imread('monalisa.jpg'), cv2.COLOR_BGR2GRAY))/255.0
im=cv2.resize(im, (28, 28))
b=np.absolute(cv2.dct(np.reshape(im, [28*28, ])))
c=np.float32(b<.5)
print(np.sum(c)/np.size(c))
plt.hist(b, bins='auto')
plt.show()

