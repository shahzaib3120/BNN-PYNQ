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

    # Encoder
    cnn = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input)

    # 1st Layer
    cnn = binary_net.Conv2DLayer(
            cnn, 
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=64, 
            filter_size=(4, 4),
            pad='valid',
            stride=(2,2),
            flip_filters=False,
            nonlinearity=lasagne.nonlinearities.identity)
    
    cnn = lasagne.layers.BatchNormLayer(cnn, epsilon=epsilon, alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(cnn, nonlinearity=activation) 

    cnn = lasagne.layers.DropoutLayer(cnn, p = 0.2)
    
    print cnn.output_shape
    # 2nd Layer
    cnn = binary_net.Conv2DLayer(
            cnn, 
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=64, 
            filter_size=(4, 4),
            pad='valid',
            stride=(2,2),
            flip_filters=False,
            nonlinearity=lasagne.nonlinearities.identity)
    
    cnn = lasagne.layers.BatchNormLayer(cnn, epsilon=epsilon, alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(cnn, nonlinearity=activation) 

    cnn = lasagne.layers.DropoutLayer(cnn, p = 0.2)

    print cnn.output_shape
    # 3rd Layer
    cnn = binary_net.Conv2DLayer(
            cnn, 
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=64, 
            filter_size=(4, 4),
            pad='valid',
            stride=(1,1),
            flip_filters=False,
            nonlinearity=lasagne.nonlinearities.identity)
    
    cnn = lasagne.layers.BatchNormLayer(cnn, epsilon=epsilon, alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(cnn, nonlinearity=activation) 

    cnn = lasagne.layers.DropoutLayer(cnn, p = 0.2)
    
    print cnn.output_shape

    cnn = lasagne.layers.flatten(cnn)

    print cnn.output_shape    
    
    # FC Layer            
    cnn = binary_net.DenseLayer(
                cnn, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=256)      
                  
    cnn = lasagne.layers.BatchNormLayer(cnn, epsilon=epsilon, alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(cnn, nonlinearity=activation)

    print cnn.output_shape

    # Deoceder
    cnn = lasagne.layers.ReshapeLayer(cnn, shape = (-1, 64, 2, 2))
    
    print cnn.output_shape
    

    # 1st Deconv Layer
    cnn = binary_net.Deconv2DLayer(
            cnn, 
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=64, 
            filter_size=(4, 4),
            crop='valid',
            stride=(2,2),
            flip_filters=False,
            nonlinearity=lasagne.nonlinearities.identity)
    
    cnn = lasagne.layers.BatchNormLayer(cnn, epsilon=epsilon, alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(cnn, nonlinearity=activation) 
    
    print cnn.output_shape

    # 2nd Deconv Layer
    cnn = binary_net.Deconv2DLayer(
            cnn, 
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=64, 
            filter_size=(4, 4),
            crop='valid',
            stride=(2,2),
            flip_filters=False,
            nonlinearity=lasagne.nonlinearities.identity)
    
    cnn = lasagne.layers.BatchNormLayer(cnn, epsilon=epsilon, alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(cnn, nonlinearity=activation) 
    
    print cnn.output_shape

    # 3rd Deconv Layer
    cnn = binary_net.Deconv2DLayer(
            cnn, 
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=1, 
            filter_size=(4, 4),
            crop='valid',
            stride=(2,2),
            flip_filters=False,
            nonlinearity=lasagne.nonlinearities.identity)
    
    cnn = lasagne.layers.BatchNormLayer(cnn, epsilon=epsilon, alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(cnn, nonlinearity=activation) 
    
    print cnn.output_shape
        
    cnn = lasagne.layers.flatten(cnn)

    print cnn.output_shape

    # Last FC layer
    cnn = binary_net.DenseLayer(
                cnn, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=num_outputs)
       
    cnn = lasagne.layers.BatchNormLayer(cnn, epsilon=epsilon, alpha=alpha)
    
    print cnn.output_shape
    
    return cnn




def genCnvInf(input, num_classes, learning_parameters):
    # ENCODER
    cnn = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input)
    cnn = lasagne.layers.Conv2DLayer(cnn, num_filters=64, filter_size=(4, 4), pad='valid', stride=(2, 2), flip_filters=False, nonlinearity=lasagne.nonlinearities.rectify)
    print cnn.output_shape
    cnn = lasagne.layers.DropoutLayer(cnn, p = 0.2)
 	
    cnn = lasagne.layers.Conv2DLayer(cnn, num_filters=64, filter_size=(4, 4), pad='valid', stride=(2, 2), flip_filters=False, nonlinearity=lasagne.nonlinearities.rectify)
    cnn = lasagne.layers.DropoutLayer(cnn, p = 0.2)
    print cnn.output_shape
    cnn = lasagne.layers.Conv2DLayer(cnn, num_filters=64, filter_size=(4, 4), pad='valid', stride=(1, 1), flip_filters=False, nonlinearity=lasagne.nonlinearities.rectify)
    cnn = lasagne.layers.DropoutLayer(cnn, p = 0.2)
    print cnn.output_shape
    cnn = lasagne.layers.flatten(cnn)
    print cnn.output_shape
    cnn = lasagne.layers.DenseLayer(cnn, nonlinearity=lasagne.nonlinearities.rectify, num_units=8)
    print cnn.output_shape
    
    # DECODER
    print "Decoder"    
    cnn = lasagne.layers.DenseLayer(cnn, nonlinearity=lasagne.nonlinearities.rectify, num_units=256)
    print cnn.output_shape
    cnn = lasagne.layers.ReshapeLayer(cnn, shape = (-1, 64, 2, 2))
    print cnn.output_shape
    cnn = lasagne.layers.Deconv2DLayer(cnn, num_filters=64, filter_size=(4,4), crop='valid', stride=(2,2), flip_filters=False, nonlinearity=lasagne.nonlinearities.rectify)
    print cnn.output_shape

    cnn = lasagne.layers.Deconv2DLayer(cnn, num_filters=64, filter_size=(4,4), crop='valid', stride=(2,2), flip_filters=False, nonlinearity=lasagne.nonlinearities.rectify)
    print cnn.output_shape

    cnn = lasagne.layers.Deconv2DLayer(cnn, num_filters=64, filter_size=(4,4), crop='valid', stride=(2,2), flip_filters=False, nonlinearity=lasagne.nonlinearities.rectify)
    print cnn.output_shape

    cnn = lasagne.layers.flatten(cnn)
    cnn = lasagne.layers.DenseLayer(cnn, nonlinearity=lasagne.nonlinearities.sigmoid, num_units=num_classes)
    print cnn.output_shape

    return cnn
