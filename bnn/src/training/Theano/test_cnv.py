from __future__ import print_function
import sys
import os
import time
import numpy as np
np.random.seed(1234)
import argparse
import theano
import theano.tensor as T
import lasagne
import binary_ops


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Testing Script')
    parser.add_argument('--dataset', '-d', default="cifar10", help='dataset to use cifar10/cifar100/mnist')
    parser.add_argument('--model', '-m', default="cnv", help='model to use resnet/lenet/inception/cnv')
    args = parser.parse_args()


    if args.dataset == 'cifar10':
        print('Loading CIFAR-10 dataset...')
        from pylearn2.datasets.cifar10 import CIFAR10
        test_set = CIFAR10(which_set="test", start=0, stop = 20)
        classes = 10
        test_set.X = np.reshape(np.subtract(np.multiply(2./255,test_set.X),1.),(-1,3,32,32))

    elif args.dataset == 'cifar100':
        print('Loading CIFAR-100 dataset...')
        from pylearn2.datasets.cifar100 import CIFAR100
        test_set = CIFAR100(which_set="test", start=0, stop = 20)
        classes = 100
        test_set.X = np.reshape(np.subtract(np.multiply(2./255,test_set.X),1.),(-1,3,32,32))

    elif args.dataset == 'mnist':
    	print('Loading MNIST dataset...')
    	from pylearn2.datasets.mnist import MNIST
    	test_set = MNIST(which_set="test", start=0, stop = 5000)
    	classes = 10
	test_set.X = 2* test_set.X.reshape(-1, 1, 28, 28) - 1.
    
    # flatten targets
    test_set.y = np.hstack(test_set.y)

    # one hot
    test_set.y = np.float32(np.eye(classes)[test_set.y])
    
    print('Building Network...') 
    
    # Prepare Theano variables for inputs and targets
    input = T.tensor4('inputs')
    target = T.matrix('targets')

    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
    	import cnv
    	cnn = cnv.genCnvInf(input, classes)

    elif args.dataset == 'mnist' and args.model == 'resnet':
    	import resnet
    	cnn = resnet.genCnvInf(input, classes)

    elif args.dataset == 'mnist' and args.model == 'lenet':
   	import lenet
   	cnn = lenet.genCnvInf(input, classes)
    elif args.dataset == 'mnist' and args.model == 'inception':
	import inception
	cnn = inception.genCnvInf(input, classes)

    test_output = lasagne.layers.get_output(cnn, deterministic=True)
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)

    val_fn = theano.function([input, target], test_err)
    
    inter_fn = theano.function([input], test_output)

    print("Loading the trained parameters and binarizing the weights...")
    if args.dataset == 'cifar10':
 	weights = '../weights/cifar10-w1a1.npz'
    elif args.dataset == 'cifar100':
	weights = '../weights/cifar100_parameters.npz'
    elif args.dataset == 'mnist' and args.model == 'resnet':
	weights = '../weights/resnet_parameters.npz'
    elif args.dataset == 'mnist' and args.model == 'lenet':
	weights = '../weights/lenet_parameters.npz'
    elif args.dataset == 'mnist' and args.model == 'inception':
	weights = '../weights/inception_parameters.npz'

    with np.load(weights) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(cnn, param_values)

    # binarize weights
    params = lasagne.layers.get_all_params(cnn)
    for param in params:
        # print(param.name)
        if param.name == "W":
            param.set_value(binary_ops.SignNumpy(param.get_value()))

    print('Testing...')
    start_time = time.time()
    test_error = val_fn(test_set.X,test_set.y)*100.
    run_time = time.time() - start_time
    print("test_error = " + str(test_error) + "%")
    print("test_acc = " + str(100-test_error) + "%")
    print("CPU Latency = "+str(run_time)+" sec")
    print("FPS = " + str(test_set.X.shape[0]/run_time))
    '''
    np.set_printoptions(threshold=np.inf)
    a = inter_fn(test_set.X)
    for i in range(20):
    	print('pred = ' + str(np.argmax(a[i])) + ' and label = ' + str(np.argmax(test_set.y[i])))# + ' error = ' + str(np.argmax(a[i]) - np.argmax(test_set.y[i])))
   	'''