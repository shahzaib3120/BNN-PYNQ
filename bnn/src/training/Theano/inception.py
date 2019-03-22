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

    out_layers = []
    inp = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input)
   
    # first conv
    cnn = binary_net.Conv2DLayer(
            inp, 
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=16, 
            filter_size=(3, 3),
            pad='same',
            flip_filters=False,
            nonlinearity=lasagne.nonlinearities.identity)

    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
    
    cnn = lasagne.layers.BatchNormLayer(cnn, epsilon=epsilon, alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(cnn, nonlinearity=activation)

    # 1x1 conv
    cnn_1x1 = binary_net.Conv2DLayer(
            cnn, 
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=32, 
            filter_size=(1, 1),
            pad='valid',
            flip_filters=False,
            nonlinearity=lasagne.nonlinearities.identity)

    cnn_1x1 = lasagne.layers.MaxPool2DLayer(cnn_1x1, pool_size=(2, 2))
    
    cnn_1x1 = lasagne.layers.BatchNormLayer(cnn_1x1, epsilon=epsilon, alpha=alpha)
                
    cnn_1x1 = lasagne.layers.NonlinearityLayer(cnn_1x1, nonlinearity=activation)
    
    out_layers.append(cnn_1x1)

    # 3x3 conv layer            
    cnn_3x3 = binary_net.Conv2DLayer(
            cnn, 
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=32, 
            filter_size=(3, 3),
            pad='same',
            flip_filters=False,
            nonlinearity=lasagne.nonlinearities.identity)

    cnn_3x3 = lasagne.layers.MaxPool2DLayer(cnn_3x3, pool_size=(2, 2))
    
    cnn_3x3 = lasagne.layers.BatchNormLayer(cnn_3x3, epsilon=epsilon, alpha=alpha)
                
    cnn_3x3 = lasagne.layers.NonlinearityLayer(cnn_3x3, nonlinearity=activation) 
    out_layers.append(cnn_3x3)
    
    # 2nd conv layer
    cnn_5x5 = binary_net.Conv2DLayer(
            cnn, 
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=32, 
            filter_size=(5, 5),
            pad='same',
            flip_filters=False,
            nonlinearity=lasagne.nonlinearities.identity)

    
    cnn_5x5 = lasagne.layers.MaxPool2DLayer(cnn_5x5, pool_size=(2, 2))
    
    cnn_5x5 = lasagne.layers.BatchNormLayer(cnn_5x5, epsilon=epsilon, alpha=alpha)
                
    cnn_5x5 = lasagne.layers.NonlinearityLayer(cnn_5x5, nonlinearity=activation) 

    out_layers.append(cnn_5x5)
    
    cnn = lasagne.layers.concat(out_layers)

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

    out_layers = []
    cnn = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input)

    cnn = lasagne.layers.Conv2DLayer(cnn, num_filters=16, filter_size=(3, 3), pad='same', flip_filters=False, nonlinearity=lasagne.nonlinearities.identity)
    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2)) 
    cnn = lasagne.layers.BatchNormLayer(cnn)
    cnn = lasagne.layers.NonlinearityLayer(cnn, nonlinearity=binary_net.SignTheano)


    cnn1x1 = lasagne.layers.Conv2DLayer(cnn, num_filters=32, filter_size=(1, 1), pad='valid', flip_filters=False, nonlinearity=lasagne.nonlinearities.identity)
    cnn1x1 = lasagne.layers.MaxPool2DLayer(cnn1x1, pool_size=(2, 2)) 
    cnn1x1 = lasagne.layers.BatchNormLayer(cnn1x1)
    cnn1x1 = lasagne.layers.NonlinearityLayer(cnn1x1, nonlinearity=binary_net.SignTheano)
    out_layers.append(cnn1x1)

    cnn3x3 = lasagne.layers.Conv2DLayer(cnn, num_filters=32, filter_size=(3, 3), pad='same', flip_filters=False, nonlinearity=lasagne.nonlinearities.identity)
    cnn3x3 = lasagne.layers.MaxPool2DLayer(cnn3x3, pool_size=(2, 2)) 
    cnn3x3 = lasagne.layers.BatchNormLayer(cnn3x3)
    cnn3x3 = lasagne.layers.NonlinearityLayer(cnn3x3, nonlinearity=binary_net.SignTheano)
    out_layers.append(cnn3x3)

    cnn5x5 = lasagne.layers.Conv2DLayer(cnn, num_filters=32, filter_size=(5, 5), pad='same', flip_filters=False, nonlinearity=lasagne.nonlinearities.identity)
    cnn5x5 = lasagne.layers.MaxPool2DLayer(cnn5x5, pool_size=(2, 2)) 
    cnn5x5 = lasagne.layers.BatchNormLayer(cnn5x5)
    cnn5x5 = lasagne.layers.NonlinearityLayer(cnn5x5, nonlinearity=binary_net.SignTheano)
    out_layers.append(cnn5x5)

    cnn = lasagne.layers.concat(out_layers)

    cnn = lasagne.layers.DenseLayer(cnn, nonlinearity=lasagne.nonlinearities.identity, num_units=512)
    cnn = lasagne.layers.BatchNormLayer(cnn)
    cnn = lasagne.layers.NonlinearityLayer(cnn,nonlinearity=binary_net.SignTheano)

    cnn = lasagne.layers.DenseLayer(cnn, nonlinearity=lasagne.nonlinearities.identity, num_units=num_classes)
    cnn = lasagne.layers.BatchNormLayer(cnn)

    return cnn

