# LCA-CS
Compressed Sensing reconstruction using locally-competitive neural networks.

## Background
Modern cameras/devices are wasteful of data, which
can be expensive to collect and transmit. It doesn’t
make sense to collect much more data than you need,
then compress it. Compressive sensing (CS) combines
sampling and compression in one step.  
  
An image x with n pixels can be transformed into a
n×1 column vector, where all the pixels are stacked
on top of each other.  
  
<p align="center">
  <b>Figure 1: The image can be transformed into a column vector.</b><br>
  <br><br>
  <img src="https://github.com/MichaelTeti/CS_LCA/blob/master/jcmeme2.jpg">
</p>
  
To take compressed samples of x, multiply it with
a random m × n matrix A, where m << n to get
compressed measurement vector b.  
  
<p align="center">
  <b>Figure 2: To sample, multiply x by a random m x n Gaussian matrix.</b><br>
  <br><br>
  <img src="https://github.com/MichaelTeti/CS_LCA/blob/master/ax%3Db.jpg">
</p>
  
To get back x from b, you have to minimize the MSE between Ax and b plus the sum of x. We add the sum of x to the minimization problem because the correct solution is sparse (i.e. has a lot of zeros). Current methods require hundreds of lines of code to solve this problem. If only there was a more simple, faster way ...  
  
Locally-Competitive Algorithms (LCAs) are a type
of dynamical neural network that can be used to
recover compressed signals using lateral inhibition
like the human visual system. Upon receiving
input, each node charges up in proportion to how
much the input resembles the appropriate stimulus.
If it charges up enough, it will "fire" and produce an
output, as well as inhibit nearby, similar nodes (red
arrows in Fig. 1) in proportion to its activation.  
  
<p align="center">
  <b>Figure 3: The network's weihts change over time to minimize the reconstruction error.</b><br>
  <br><br>
  <img src="https://github.com/MichaelTeti/LCA_Sparse_Coding_WadingBirds/blob/master/LCA1.jpg">
</p>

We send A, x, and b to the network and, over time, it will settle on a sparse approximation of the original image.  

## Algorithm
lambda = 4.0    
h = 0.005  
u = zeros(n, 1)  
While MSE is above some value:  
u = u + h x (A' x (b − A*x*) − u − *x*)  
x=(u-sign(u).*(lambda)).*(abs(u)>(lambda))  
  
The variable u is the input layer, h is a scale constant, and lambda is the threshold. As can be seen, the algorithm is extremely simple
and efficient. Furthermore, it is a vectorized implementation,
which enables the use of GPUs, making
it much faster than alternative methods.
The first line receives the input and minimizes the
mean-squared error between Ax and b, while also
causing the inhibition of nearby nodes.
The second line is the activation function of each
node, which causes nodes with activations below
threshold to output nothing. In all, the network outputs
a sparse solution that approximates, or sometimes
equals, the original image.

## Results
### Reconstructions of an image with different sampling rates.
<p align="center">
  <b>Figure 3: The network's weihts change over time to minimize the reconstruction error.</b><br>
  <br><br>
  <img src="https://github.com/MichaelTeti/CS_LCA/blob/master/circles.png">
</p>
### Reconstruction of a natural image. 
<p align="center">
  <b>Figure 3: The network's weihts change over time to minimize the reconstruction error.</b><br>
  <br><br>
  <img src="https://github.com/MichaelTeti/CS_LCA/blob/master/mona_lca_l1_l2_orig.png">
</p>

The locally-competitive algorithm method is able to approximately reconstruct the signal in almost 9 times less time than the current state-of-the-art method. 

## Further Reading
- [LCA](https://www.google.com/patents/US7783459) 
- [Compressed Sensing](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1580791)
- [Single-pixel camera](https://www.youtube.com/watch?v=RvMgVv-xZhQ&t=2748s)

