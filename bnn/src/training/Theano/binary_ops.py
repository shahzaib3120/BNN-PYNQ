import time

import numpy as np
import theano
import theano.tensor as T
import lasagne


def SignTheano(x):
    return T.cast(2.*T.ge(x,0)-1., theano.config.floatX)


def SignNumpy(x):
    return np.float32(2.*np.greater_equal(x,0)-1.)


class Conv2DLayer(lasagne.layers.Conv2DLayer):

    def __init__(self, incoming, num_units, kernel="theano", **kwargs):
        
        self.kernel = kernel
        super(DenseLayer, self).__init__(incoming, num_units, **kwargs)
            
    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)
        
        
        if self.kernel == "theano":
            activation = T.dot(input, self.W)
        
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)

class DenseLayer(lasagne.layers.DenseLayer):

    def __init__(self, incoming, num_units, kernel="theano", **kwargs):
        
        self.kernel = kernel
        super(DenseLayer, self).__init__(incoming, num_units, **kwargs)
            
    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)
        
        
        if self.kernel == "theano":
            activation = T.dot(input, self.W)
        
        
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)