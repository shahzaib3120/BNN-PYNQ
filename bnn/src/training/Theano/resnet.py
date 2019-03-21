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

import lasagne
import binary_net
from lasagne.layers import ElemwiseSumLayer

def genCnv(input, num_outputs, learning_parameters):
    # A function to generate the cnv network topology which matches the overlay for the Pynq board.
    # WARNING: If you change this file, it's likely the resultant weights will not fit on the Pynq overlay.
    stochastic = False
    binary = True
    H = 1
    activation = binary_net.binary_tanh_unit
    W_LR_scale = learning_parameters.W_LR_scale
    epsilon = learning_parameters.epsilon
    alpha = learning_parameters.alpha

    inp = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input)
   
    # first conv
    cnn = binary_net.Conv2DLayer(
            inp, 
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=64, 
            filter_size=(3, 3),
            pad='same',
            flip_filters=False,
            nonlinearity=lasagne.nonlinearities.identity)

    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
    
    cnn = lasagne.layers.BatchNormLayer(cnn, epsilon=epsilon, alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(cnn, nonlinearity=activation)

    residual = cnn


    # conv 1 in Res block
    cnn = binary_net.Conv2DLayer(
            cnn, 
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=64, 
            filter_size=(3, 3),
            pad='same',
            flip_filters=False,
            nonlinearity=lasagne.nonlinearities.identity)

    cnn = lasagne.layers.BatchNormLayer(cnn, epsilon=epsilon, alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(cnn, nonlinearity=activation)


    # special conv 3 is Res block 
    cnn = binary_net.Conv2DLayer(
            cnn, 
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=64, 
            filter_size=(3, 3),
            pad='same',
            flip_filters=False,
            nonlinearity=lasagne.nonlinearities.identity)


    cnn =ElemwiseSumLayer([residual, cnn], coeffs=1)

    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
    
    cnn = lasagne.layers.BatchNormLayer(cnn, epsilon=epsilon, alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(cnn, nonlinearity=activation)


    # FC layer 1            
    cnn = binary_net.DenseLayer(
                cnn, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=512)      
                  
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation) 

    # FC layer 2        
    cnn = binary_net.DenseLayer(
                cnn, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=num_outputs)

             
    cnn = lasagne.layers.BatchNormLayer(cnn, epsilon=epsilon, alpha=alpha)
    
    return cnn

def genCnvInf(input, num_classes):
    
    cnn = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input)

    cnn = lasagne.layers.Conv2DLayer(cnn, num_filters=64, filter_size=(3, 3), pad='same', flip_filters=False, nonlinearity=lasagne.nonlinearities.identity)
    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2)) 
    cnn = lasagne.layers.BatchNormLayer(cnn)
    cnn = lasagne.layers.NonlinearityLayer(cnn, nonlinearity=binary_net.SignTheano)

    residual = cnn

    cnn = lasagne.layers.Conv2DLayer(cnn, num_filters=64, filter_size=(3, 3), pad='same', flip_filters=False, nonlinearity=lasagne.nonlinearities.identity)
    cnn = lasagne.layers.BatchNormLayer(cnn)
    cnn = lasagne.layers.NonlinearityLayer(cnn, nonlinearity=binary_net.SignTheano)

    cnn = lasagne.layers.Conv2DLayer(cnn, num_filters=64, filter_size=(3, 3), pad='same', flip_filters=False, nonlinearity=lasagne.nonlinearities.identity)
    cnn = ElemwiseSumLayer([residual, cnn], coeffs=1)
    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2)) 
    cnn = lasagne.layers.BatchNormLayer(cnn)
    cnn = lasagne.layers.NonlinearityLayer(cnn, nonlinearity=binary_net.SignTheano)

    cnn = lasagne.layers.DenseLayer(cnn, nonlinearity=lasagne.nonlinearities.identity, num_units=512)
    cnn = lasagne.layers.BatchNormLayer(cnn)
    cnn = lasagne.layers.NonlinearityLayer(cnn,nonlinearity=binary_net.SignTheano)

    cnn = lasagne.layers.DenseLayer(cnn, nonlinearity=lasagne.nonlinearities.identity, num_units=num_classes)
    cnn = lasagne.layers.BatchNormLayer(cnn)

    return cnn
