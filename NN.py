# This is a pedagogically and visually better layout of a neural network, atleast that's what I think - Prashaanth

import numpy as np
import pandas as pd

# _d signifies dimensions
# self.pushed is the output of the forward pass

# DENSE CLASS DEFINED >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class Dense:
    def __init__(self, in_d, n_d):
        # std mean based initial weight values advised
        std_dev = np.sqrt(2/in_d)
        
        # the below dimensions are set with respect to having
        # the input dimensions synonymous to a neural network input
        # with vertical inputs.
        # inputs will be matrix of (in_d, batch_size)

        # hence, neural feed forward computation:
        # # (n_d, in_d).(in_d, batch_size) = (n_d, batch_size) and + (n_d, 1) = (n_d, batch_size)

        self.weights = np.random.normal(0, std_dev, (n_d, in_d))
        # self.weights = 0.01 * np.random.randn(n_d, in_d)
        self.biases = np.zeros((n_d, 1))

    def forward(self, inputs):
        self.inputs = inputs # size => (in_d, batch_size)

        #                    (n_d, in_d)   (in_d, batch_size)   (n_d, 1)
        self.pushed = np.dot(self.weights, self.inputs) + self.biases        # (n_d, batch_size)
        return self.pushed
    
    def backward(self, dvalues):
        # dvalues are incoming in shape (n_d, batch_size)
        self.dweights = np.dot(self.inputs, dvalues.T).T            # (n_d, in_d)
        self.dbiases =  np.sum(dvalues, axis=1, keepdims=True)      # (n_d, 1)
        self.dinputs =  np.dot(dvalues.T, self.weights).T           # (in_d, batch_size), to be sent to prev layer

# ACTVATION CLASS - ReLU - DEFINED >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class Activation_ReLU:
    def forward(self, z):
        self.z = z                  # (n_d, batch_size)
        self.a = np.maximum(0, z)   # (n_d, batch_size)
        return self.a

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.z <= 0] = 0

# ACTIVATION CLASS - SOFTMAX - DEFINED >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class Activation_Softmax:
    def forward(self, z):
        self.z = z      # (n_d, batch_size)
        exps = np.exp(z - np.max(z, axis=0, keepdims=True))
        self.a = exps / np.sum(exps, axis=0, keepdims=True)     # (n_d, batch_size)
        return self.a

# LOSS CLASS - CCEL - DEFINED >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class Loss_CrossCategoricalEntropy:
    def forward(self, y_cap, y):
        self.y_cap = y_cap    # (n_d, batch_size)
        self.loss_val = -np.mean(np.sum(y * np.log(y_cap + 1e-12), axis=0))      # sum of all loss accross batches is total loss
        return self.loss_val

# SOFTMAX + ACTIVATION CLASS DEFINED >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class Activation_Softmax_Loss_CCE:
    
    def __init__(self):
        self.activation_func = Activation_Softmax()
        self.loss_func = Loss_CrossCategoricalEntropy()
    
    def forward(self, z, y):
        self.z = z      # (n_d, batch_size)
        self.y = y      # (n_d, batch_size)
        self.softed = self.activation_func.forward(self.z)      # (n_d, batch_size)
        self.loss_val = self.loss_func.forward(self.softed, y)  # (scalar)
        return self.loss_val

    def backward(self):
        self.dL_dZ = self.softed - self.y   # (n_d, batch_size)
        batch_size = self.y.shape[1]
        self.dL_dZ /= batch_size

# SGD optimizer class >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Atlas (Sorta)
# learning rate decay - reducing the initially high learning rate based on the number of updations so far to the weights and biases
# momentum - using velocity model to sustain training without much oscillations in gradients
# gradient scaling, introduced in ADAGRAD - used to calculate suitable learning rates for different weights
# gradient scaling with rho - used to stop continuously growing gradient scalers (divisors), ideally stopping dead neurons
# Exponential Moving Averages, introduced in ADAM - used to smooth out the gradients (like momentum) and used to smooth out the gradient scaler/divisor (like RMSprop)

class Optimizer_SGD:
    def __init__(
            self,
            alpha    = 0.001,
            decay    = 0,
            epsilon  = 1e-7,    # unused
            momentum = 0.9,
            rho      = 0.999    # unused
            
        ):

        self.alpha = alpha
        self.current_alpha = alpha
        self.decay = decay
        self.momentum = momentum
        self.iterations = 0

    def pre_update_params(self):
        if self.decay:
            self.current_alpha = self.alpha / (1. + self.decay * self.iterations)

    def update_params(self, layer):

        if self.momentum:
            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = self.momentum * layer.weight_momentums - self.current_alpha * layer.dweights
            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums - self.current_alpha * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_alpha * layer.dweights
            bias_updates = -self.current_alpha * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1


# ADAGRAD optimizer class >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>?????????>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class Optimizer_ADAGRAD:
    def __init__(
            self,
            alpha    = 0.001,
            decay    = 0,
            epsilon  = 1e-7,
            momentum = 0.9,     # unused
            rho      = 0.999    # unused
            
        ):

        self.alpha = alpha
        self.current_alpha = alpha
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def pre_update_params(self):
        if self.decay:
            self.current_alpha = self.alpha / (1. + self.decay * self.iterations)

    def update_params(self, layer):

        # If layer does not contain cache arrays, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        # Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights += -self.current_alpha * layer.dweights / (np.sqrt(layer.weight_cache + self.epsilon))
        layer.biases += -self.current_alpha  * layer.dbiases  / (np.sqrt( layer.bias_cache  + self.epsilon))

    def post_update_params(self):
        self.iterations += 1


# RMSprop optimizer class >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>?????????>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class Optimizer_RMSprop:
    def __init__(
            self,
            alpha    = 0.001,
            decay    = 0,
            epsilon  = 1e-7,
            momentum = 0.9,     # unused
            rho      = 0.999
        ):

        self.alpha = alpha
        self.current_alpha = alpha
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def pre_update_params(self):
        if self.decay:
            self.current_alpha = self.alpha / (1. + self.decay * self.iterations)

    def update_params(self, layer):

        # If layer does not contain cache arrays, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache = self.rho*layer.weight_cache + (1-self.rho)*layer.dweights**2
        layer.bias_cache   = self.rho*layer.bias_cache   + (1-self.rho)*layer.dbiases**2

        # Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights += -self.current_alpha * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases  += -self.current_alpha * layer.dbiases  / (np.sqrt(layer.bias_cache)   + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


# ADAM optimizer class >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>?????????>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class Optimizer_ADAM:
    def __init__(
            self,
            alpha    = 0.001,
            decay    = 0,
            epsilon  = 1e-7,
            momentum = 0.9,
            rho      = 0.999
            
        ):

        self.alpha          = alpha
        self.current_alpha  = alpha
        self.decay          = decay
        self.iterations     = 0
        self.epsilon        = epsilon
        self.rho            = rho
        self.momentum       = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_alpha = self.alpha / (1. + self.decay * self.iterations)

    def update_params(self, layer):

        # If layer does not contain cache arrays, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums   = np.zeros_like(layer.biases)

            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache   = np.zeros_like(layer.biases)

        # Update momentum with current gradients
        layer.weight_momentums = self.momentum*layer.weight_momentums + (1 - self.momentum)*layer.dweights
        layer.bias_momentums   = self.momentum*layer.bias_momentums   + (1 - self.momentum)*layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / (1 - self.momentum ** (self.iterations + 1))
        bias_momentums_corrected   = layer.bias_momentums   / (1 - self.momentum ** (self.iterations + 1))

        # Update cache with squared current gradients
        layer.weight_cache = self.rho*layer.weight_cache + (1-self.rho)*layer.dweights**2
        layer.bias_cache   = self.rho*layer.bias_cache   + (1-self.rho)*layer.dbiases**2

        weight_cache_corrected = layer.weight_cache / (1 - self.rho ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.rho ** (self.iterations + 1))

        # Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights += -self.current_alpha * weight_momentums_corrected / (np.sqrt(weight_cache_corrected + self.epsilon))
        layer.biases  += -self.current_alpha * bias_momentums_corrected   / (np.sqrt(bias_cache_corrected   + self.epsilon))

    def post_update_params(self):
        self.iterations += 1
        

# NEURAL NETWORK CLASS DEFINED >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class NN:
    def __init__(
            self,
            in_d = 784,
            d1   = 64,
            d2   = 64,
            d3   = 64,
            o_d  = 10
    ):
        # this is where the definition starts

        # neural layers defined:
        # each layer defined by number of inputs and number of neurons
        # input shaped as (in_d, batch_size) , will be followed

        self.l1  = Dense(in_d, d1)
        self.al1 = Activation_ReLU()

        self.l2  = Dense(d1  , d2)
        self.al2 = Activation_ReLU()

        self.l3  = Dense(d2  , d3)
        self.al3 = Activation_ReLU()

        self.ol  = Dense(d3 , o_d)
        
        self.activation_loss_l = Activation_Softmax_Loss_CCE()

        # the optimizer for the NN class must be specified in the place where you initiate the nn class instance

    def forward(self, inputs, y):
        assert inputs.shape[0] == self.l1.weights.shape[1]

        a0 = inputs                                 # size => (in_d, batch_size)
        
        self.l1.forward(a0)                         # size => (n_d, batch_size)
        self.al1.forward(self.l1.pushed)            # size => (n_d, batch_size)

        self.l2.forward(self.al1.a)                 # size => (n_d, batch_size)
        self.al2.forward(self.l2.pushed)            # size => (n_d, batch_size)

        self.l3.forward(self.al2.a)                 # size => (n_d, batch_size)
        self.al3.forward(self.l3.pushed)            # size => (n_d, batch_size)

        self.ol.forward(self.al3.a)                 # size => (o_d, batch_size)
        self.activation_loss_l.forward(self.ol.pushed, y) # size => (o_d, batch_size)

        self.loss_val = self.activation_loss_l.loss_val

        return self.loss_val
    
    # we define a loss function and then try to minimize it
    # the loss is in essence a multivariable func
    # dependent on all weights and biases, and we have to
    # find how each of them affects the loss function, through gradients
    # gradients are then multiplied with a learning rate and then
    # subtracted from the actual weights and biases

    def backward(self):
        self.activation_loss_l.backward()
        self.ol.backward(self.activation_loss_l.dL_dZ)
        self.al3.backward(self.ol.dinputs)
        self.l3.backward(self.al3.dinputs)
        self.al2.backward(self.l3.dinputs)
        self.l2.backward(self.al2.dinputs)
        self.al1.backward(self.l2.dinputs)
        self.l1.backward(self.al1.dinputs)

    def update(self):
        self.optimizer.pre_update_params()

        self.optimizer.update_params(self.l1)
        self.optimizer.update_params(self.l2)
        self.optimizer.update_params(self.l3)
        self.optimizer.update_params(self.ol)

        self.optimizer.post_update_params()