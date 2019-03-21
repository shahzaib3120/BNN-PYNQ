from __future__ import print_function
import sys
import os
import time
import numpy as np
np.random.seed(1234) # for reproducibility?
# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1') 
import theano
import theano.tensor as T
import lasagne
import cPickle as pickle
import gzip
import binary_net
from pylearn2.datasets.mnist import MNIST
from pylearn2.utils import serial
from collections import OrderedDict
import lfc

if __name__ == "__main__":

    #image_no = 0
    batch_size = 10000
    
    #print("Image No. = "+str(image_no))
    print("batch_size = "+str(batch_size))

    print('Loading MNIST dataset...')
    
    #test_set = MNIST(which_set= 'test', start=(0 + image_no), stop = (image_no + 1), center = False)
    test_set = MNIST(which_set= 'test', start=0, stop = batch_size, center = False)

    # Inputs in the range [-1,+1]
    # test_set.X = 2* test_set.X.reshape(-1, 784) - 1. 
    test_set.X = 2* test_set.X.reshape(-1, 1, 28, 28) - 1.

    
    # binarize the inputs
    test_set.X = np.where(test_set.X < 0, -1, 1).astype(theano.config.floatX) 

    # flatten targets
    #test_set.y = np.hstack(test_set.y)
    test_set.y = test_set.y.reshape(-1) 

    # one hot
    #test_set.y = np.float32(np.eye(10)[test_set.y])
    
    #print(test_set.X.shape)
    #print(test_set.y.shape)
    #print(test_set.X)
    #print(test_set.y)
    #exit(0)

    print('Building MLP...') 
    
    # Prepare Theano variables for inputs and targets
    input = T.tensor4('inputs')
    target = T.vector('targets')

    mlp = lfc.genLfcInf(input, 10)

    test_output = lasagne.layers.get_output(mlp, deterministic=True)
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), target),dtype=theano.config.floatX)

    val_fn = theano.function([input, target], test_err)
    
    print("Loading the trained parameters and binarizing the weights...")
    
    with np.load('mnist-w1a1.npz') as f:
    #with np.load('../weights/mnist-w1a2.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(mlp, param_values)

    params = lasagne.layers.get_all_params(mlp)
    for param in params:
        if param.name == "W":
            param.set_value(binary_net.SignNumpy(param.get_value()))

    print('Testing...')

    start_time = time.time()
    test_error = val_fn(test_set.X,test_set.y)*100.
    run_time = time.time() - start_time
    print("test_error = " + str(test_error) + "%")
    print("test_acc = " + str(100-test_error) + "%")
    print("CPU Latency = "+str(run_time)+" sec")
    print("FPS = "+ str(10000/run_time))
