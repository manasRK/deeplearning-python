
"""
Variable        	Definition
X	        Input dataset matrix where each row is a training example
y	        Output dataset matrix where each row is a training example
l0	        First Layer of the Network, specified by the input data
l1	        Second Layer of the Network, otherwise known as the hidden layer
syn0	        First layer of weights, Synapse 0, connecting l0 to l1.
*	        Elementwise multiplication, so two vectors of equal size are multiplying corresponding values 1-to-1 to generate a final          vector of identical size.
-	        Elementwise subtraction, so two vectors of equal size are subtracting corresponding values 1-to-1 to generate a final             vector of identical size.
x.dot(y)	If x and y are vectors, this is a dot product. If both are matrices, it's a matrix-matrix multiplication. If only one is a         matrix, then it's vector matrix multiplication.
"""

"""
This tutorial teaches backpropagation via a very simple toy example, a short python implementation. 
Thanks to @iamtrask, from http://iamtrask.github.io/2015/07/12/basic-python-network/
"""

"""
This imports numpy, which is a linear algebra library. This is our only dependency.
"""
import numpy as np

""" 
 This is our "nonlinearity". It can be several kinds of functions, this nonlinearity maps a function called a "sigmoid". 
 A sigmoid function maps any value to a value between 0 and 1. We use it to convert numbers to probabilities. It also has 
 several other desirable properties for training neural networks. Notice that this function can also generate the derivative 
 of a sigmoid (when deriv=True). One of the desirable properties of a sigmoid function is that its output can be used to create 
 its derivative. If the sigmoid's output is a variable "out", then the derivative is simply out * (1-out). This is very efficient. 
 If you're unfamililar with derivatives, just think about it as the slope of the sigmoid function at a given point.
"""

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
"""
This initializes our input dataset as a numpy matrix. Each row is a single "training example". 
Each column corresponds to one of our input nodes. Thus, we have 3 input nodes to the network and 4 training examples.
"""

# input dataset
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])

"""
This initializes our output dataset. In this case, I generated the dataset horizontally (with a single row and 4 columns) for space. 
".T" is the transpose function. After the transpose, this y matrix has 4 rows with one column. Just like our input, each row is 
a training example, and each column (only one) is an output node. So, our network has 3 inputs and 1 output.
"""
# output dataset            
y = np.array([[0,0,1,1]]).T

"""
It's good practice to seed your random numbers. Your numbers will still be randomly distributed, but they'll be randomly 
distributed in exactly the same way each time you train. This makes it easier to see how your changes affect the network.
"""
# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

"""
This is our weight matrix for this neural network. It's called "syn0" to imply "synapse zero". Since we only have 2 layers 
(input and output), we only need one matrix of weights to connect them. Its dimension is (3,1) because we have 3 inputs and 
1 output. Another way of looking at it is that l0 is of size 3 and l1 is of size 1. Thus, we want to connect every node in l0 
to every node in l1, which requires a matrix of dimensionality (3,1). :) 

Also notice that it is initialized randomly with a mean of zero. There is quite a bit of theory that goes into weight 
initialization For now, just take it as a best practice that it's a good idea to have a mean of zero in weight initialization. 

Another note is that the "neural network" is really just this matrix. We have "layers" l0 and l1 but they are transient values 
based on the dataset. We don't save them. All of the learning is stored in the syn0 matrix.
"""
# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1

"""
This begins our actual network training code. This for loop "iterates" multiple times over the training code to optimize our 
network to the dataset.
"""

for iter in xrange(10000):
    """
    Since our first layer, l0, is simply our data. We explicitly describe it as such at this point. Remember that X contains 4 
    training examples (rows). We're going to process all of them at the same time in this implementation. This is known as 
    "full batch" training. Thus, we have 4 different l0 rows, but you can think of it as a single training example if you want. 
    It makes no difference at this point. (We could load in 1000 or 10,000 if we wanted to without changing any of the code).
    """
    # forward propagation
    l0 = X
    """
    This is our prediction step. Basically, we first let the network "try" to predict the output given the input. We will then 
    study how it performs so that we can adjust it to do a bit better for each iteration. This line contains 2 steps. 
    
    The first matrix multiplies l0 by syn0. The second passes our output through the sigmoid function. 
    Consider the dimensions of each: (4 x 3) dot (3 x 1) = (4 x 1) 
    
    Matrix multiplication is ordered, such the dimensions in the middle of the equation must be the same. The final matrix 
    generated is thus the number of rows of the first matrix and the number of columns of the second matrix.
    
    Since we loaded in 4 training examples, we ended up with 4 guesses for the correct answer, a (4 x 1) matrix. 
    Each output corresponds with the network's guess for a given input. Perhaps it becomes intuitive why we could 
    have "loaded in" an arbitrary number of training examples. The matrix multiplication would still work out. :)
    """
    l1 = nonlin(np.dot(l0,syn0))
    
    """
    So, given that l1 had a "guess" for each input. We can now compare how well it did by subtracting the true answer (y) from 
    the guess (l1). l1_error is just a vector of positive and negative numbers reflecting how much the network missed.
    """
    # how much did we miss?
    l1_error = y - l1
    
    """
    Now we're getting to the good stuff! This is the secret sauce! There's a lot going on in this line, so let's further break 
    it into two parts.
    
    CODE: nonlin(l1,True)
    
    If l1 represents these three dots, the code above generates the slopes of the lines below. Notice that very high values 
    such as x=2.0 (green dot) and very low values such as x=-1.0 (purple dot) have rather shallow slopes. The highest slope 
    you can have is at x=0 (blue dot). This plays an important role. Also notice that all derivatives are between 0 and 1.
    
    CODE: l1_error * nonlin(l1,True)
    
    There are more "mathematically precise" ways than "The Error Weighted Derivative" but I think that this captures the 
    intuition. l1_error is a (4,1) matrix. nonlin(l1,True) returns a (4,1) matrix. What we're doing is multiplying them 
    "elementwise". This returns a (4,1) matrix l1_delta with the multiplied values. When we multiply the "slopes" by the error, 
    we are reducing the error of high confidence predictions. Look at the sigmoid picture again! If the slope was really 
    shallow (close to 0), then the network either had a very high value, or a very low value. This means that the network 
    was quite confident one way or the other. However, if the network guessed something close to (x=0, y=0.5) then it isn't 
    very confident. We update these "wishy-washy" predictions most heavily, and we tend to leave the confident ones alone 
    by multiplying them by a number close to 0.
    """

    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1,True)

    """
    In this training example, we're all setup to update our weights. Let's update the far left weight (9.5).
    CODE: weight_update = input_value * l1_delta
    
    For the far left weight, this would multiply 1.0 * the l1_delta. Presumably, this would increment 9.5 ever so slightly. 
    Why only a small ammount? Well, the prediction was already very confident, and the prediction was largely correct. A 
    small error and a small slope means a VERY small update. Consider all the weights. It would ever so slightly increase 
    all three.
    
    It computes the weight updates for each weight for each training example, sums them, and updates the weights, 
    all in a simple line. 
    """
    # update weights
    syn0 += np.dot(l0.T,l1_delta)

print "Output After Training:"
print l1
