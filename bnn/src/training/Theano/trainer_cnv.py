#BSD 3-Clause License
#=======
#
#Copyright (c) 2017, Xilinx Inc.
#All rights reserved.
#
#Based Matthieu Courbariaux's CIFAR-10 example code
#Copyright (c) 2015-2016, Matthieu Courbariaux
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the copyright holder nor the names of its 
#      contributors may be used to endorse or promote products derived from 
#      this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
#EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
#DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import print_function

import sys
import os
import time
import subprocess

import numpy as np
np.random.seed(1234) # for reproducibility?

# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1') 
import theano
import theano.tensor as T

import lasagne
import argparse

import cPickle as pickle
import gzip

import binary_net

from pylearn2.datasets.zca_dataset import ZCA_Dataset   
from pylearn2.utils import serial

from collections import OrderedDict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cifar training')
    parser.add_argument('--dataset', '-d', default="cifar10", help='dataset to use cifar10/cifar100/mnist')
    parser.add_argument('--model', '-m', default="cnv", help='model to use resnet/lenet/inception/cnv')
    args = parser.parse_args()


    learning_parameters = OrderedDict()
    # BN parameters
    batch_size = 50
    print("batch_size = "+str(batch_size))
    # alpha is the exponential moving average factor
    learning_parameters.alpha = .1
    print("alpha = "+str(learning_parameters.alpha))
    learning_parameters.epsilon = 1e-4
    print("epsilon = "+str(learning_parameters.epsilon)) 
    # W_LR_scale = 1.    
    learning_parameters.W_LR_scale = "Glorot" # "Glorot" means we are using the coefficients from Glorot's paper
    print("W_LR_scale = "+str(learning_parameters.W_LR_scale))   
    # Training parameters
    num_epochs = 10
    print("num_epochs = "+str(num_epochs))   
    # Decaying LR 
    LR_start = 0.001
    print("LR_start = "+str(LR_start))
    LR_fin = 0.0000003
    print("LR_fin = "+str(LR_fin))
    LR_decay = (LR_fin/LR_start)**(1./num_epochs)
    print("LR_decay = "+str(LR_decay))
    shuffle_parts = 1
    print("shuffle_parts = "+str(shuffle_parts))
    	  
    
    if args.dataset == 'cifar10':
    	print('Loading CIFAR-10 dataset...')
    	from pylearn2.datasets.cifar10 import CIFAR10
        train_set = CIFAR10(which_set="train",start=0,stop = 45000)
        valid_set = CIFAR10(which_set="train",start=45000,stop = 50000)
        test_set = CIFAR10(which_set="test")
        classes = 10
        save_path = "../weights/cifar10-w1a1.npz"
        print("save_path = "+str(save_path))
        train_set.X = np.reshape(np.subtract(np.multiply(2./255.,train_set.X),1.),(-1,3,32,32))
	valid_set.X = np.reshape(np.subtract(np.multiply(2./255.,valid_set.X),1.),(-1,3,32,32))
	test_set.X = np.reshape(np.subtract(np.multiply(2./255.,test_set.X),1.),(-1,3,32,32))

    elif args.dataset == 'cifar100':
    	print('Loading CIFAR-100 dataset...')
    	from pylearn2.datasets.cifar100 import CIFAR100
    	pylearn_path = os.environ['PYLEARN2_DATA_PATH']
    	path = os.path.join(pylearn_path,'cifar100', 'cifar-100-python')
    	if not os.path.exists(path):
    		cmd = subprocess.call('scripts/download_cifar100.sh')
    	train_set = CIFAR100(which_set="train",start=0,stop = 45000)
        valid_set = CIFAR100(which_set="train",start=45000,stop = 50000)
        test_set = CIFAR100(which_set="test")
        classes = 100
        save_path = "../weights/cifar100_parameters.npz"
        print("save_path = "+str(save_path))
        train_set.X = np.reshape(np.subtract(np.multiply(2./255.,train_set.X),1.),(-1,3,32,32))
	valid_set.X = np.reshape(np.subtract(np.multiply(2./255.,valid_set.X),1.),(-1,3,32,32))
	test_set.X = np.reshape(np.subtract(np.multiply(2./255.,test_set.X),1.),(-1,3,32,32))

    elif args.dataset == 'mnist':
    	print('Loading MNIST dataset...')
    	from pylearn2.datasets.mnist import MNIST
        train_set = MNIST(which_set= 'train', start=0, stop = 50000, center = False)
	valid_set = MNIST(which_set= 'train', start=50000, stop = 60000, center = False)
	test_set = MNIST(which_set= 'test', center = False)
	classes = 10
        if args.model == 'resnet':
    	   save_path = "../weights/resnet_parameters.npz"
    	   print("save_path = "+str(save_path))
        elif args.model == 'lenet':
            save_path = "../weights/lenet_parameters.npz"
            print("save_path = "+str(save_path))
        elif args.model == 'inception':
	    save_path = "../weights/inception_parameters.npz"
            print("save_path = "+str(save_path))
	train_set.X = 2* train_set.X.reshape(-1, 1, 28, 28) - 1.
	valid_set.X = 2* valid_set.X.reshape(-1, 1, 28, 28) - 1.
	test_set.X = 2* test_set.X.reshape(-1, 1, 28, 28) - 1.
 
    # flatten targets
    train_set.y = np.hstack(train_set.y)
    valid_set.y = np.hstack(valid_set.y)
    test_set.y = np.hstack(test_set.y)
    
    # Onehot the targets
    train_set.y = np.float32(np.eye(classes)[train_set.y])    
    valid_set.y = np.float32(np.eye(classes)[valid_set.y])
    test_set.y = np.float32(np.eye(classes)[test_set.y])
    
    # for hinge loss
    train_set.y = 2* train_set.y - 1.
    valid_set.y = 2* valid_set.y - 1.
    test_set.y = 2* test_set.y - 1.

    print('Building Network...') 
    
    # Prepare Theano variables for inputs and targets
    input = T.tensor4('inputs')
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)


    if args.model == 'cnv':
    	import cnv
    	cnn = cnv.genCnv(input, classes, learning_parameters)
    elif args.model == 'resnet':
    	import resnet
    	cnn = resnet.genCnv(input, classes, learning_parameters)  
    elif args.model == 'lenet':
        import lenet
        cnn = lenet.genCnv(input, classes, learning_parameters)
    elif args.model == 'inception':
    	import inception
    	cnn = inception.genCnv(input, classes, learning_parameters) 

    train_output = lasagne.layers.get_output(cnn, deterministic=False)
    
    # squared hinge loss
    loss = T.mean(T.sqr(T.maximum(0.,1.-target*train_output)))
    
    # W updates
    W = lasagne.layers.get_all_params(cnn, binary=True)
    W_grads = binary_net.compute_grads(loss,cnn)
    updates = lasagne.updates.adam(loss_or_grads=W_grads, params=W, learning_rate=LR)
    updates = binary_net.clipping_scaling(updates,cnn)
    
    # other parameters updates
    params = lasagne.layers.get_all_params(cnn, trainable=True, binary=False)
    updates = OrderedDict(updates.items() + lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR).items())

    test_output = lasagne.layers.get_output(cnn, deterministic=True)
    test_loss = T.mean(T.sqr(T.maximum(0.,1.-target*test_output)))
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)
    
    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary) 
    # and returning the corresponding training loss:
    train_fn = theano.function([input, target, LR], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input, target], [test_loss, test_err])

    print('Training...')
    
    binary_net.train(
            train_fn,val_fn,
            cnn,
            batch_size,
            LR_start,LR_decay,
            num_epochs,
            train_set.X,train_set.y,
            valid_set.X,valid_set.y,
            test_set.X,test_set.y,
            save_path=save_path,
            shuffle_parts=shuffle_parts)
