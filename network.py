import numpy as np 
import itertools
from activations import Sigmoid, Softmax, ReLU, Tanh, Identity
from loss import CrossEntropy, MSE

class Layer(object):
    def __init__(self) -> None:
        super().__init__()

    """Base class for all layers"""
    def get_output(self, X):
        """Returns layer output"""
        pass 

    def get_input_grad(self, Y, output_grad=None, T=None):
        """Computes gradient at the input of this layer"""
        pass


class ActivationLayer(Layer):
    """The activation layer applies an activation function to its input"""
    def __init__(self, activation='sigmoid') -> None:
        super().__init__()
        if activation == 'Identity':
            self.activation = Identity()
        elif activation == 'ReLU':
            self.activation = ReLU()
        elif activation == 'tanh':
            self.activation = Tanh()
        else:  # default activation
            self.activation = Sigmoid()
    
    def get_output(self, X):
        """Return the output of this layer"""
        return self.activation.call(X)
    
    def get_input_grad(self, Y, output_grad=None, T=None):
        """Return the gradient of the loss with respect to the input of this layer"""
        return np.multiply(output_grad, self.activation.deriv(Y)) / Y.shape[0]



class LinearLayer(Layer):
    """The linear layer performs a linear transformation to its input"""
    def __init__(self, in_features, out_features, initialisation) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.initialisation = initialisation 
        self.weight_initialisation()
        self.bias = np.zeros(out_features)
        self.ut_w = 0  # momentum gradient descent
        self.ut_b = 0  # momentum gradient descent
        self.mt_w = 0  # RMSprop, adam and nadam
        self.mt_b = 0 # RMSprop, adam and nadam
        self.vt_w = 0  # adam and nadam
        self.vt_b = 0  # adam and nadam
    
    def weight_initialisation(self):
        # seed the random number generator
        np.random.seed(32)
        if (self.initialisation == 'random'):
            self.weights = np.random.normal(loc=0, scale=1.0, size=(self.in_features, self.out_features)) * 0.1

        elif (self.initialisation == 'xavier'):
            upper_bound = np.sqrt(6.0 / (self.in_features + self.out_features))
            lower_bound = -np.sqrt(6.0 / (self.in_features + self.out_features))
            self.weights = np.random.uniform(lower_bound, upper_bound, (self.in_features, self.out_features))

        # else:   # default initialisation
        #     self.weights = np.random.randn(self.in_features, self.out_features) * np.sqrt(2 / self.in_features)
    
    def get_output(self, X):
        """Return the output of this layer"""
        return (X @ self.weights) + self.bias
    
    def get_params_grad(self, X, output_grad):
        """Return a list of gradients over the parameters of this layer"""
        self.JW = X.T @ output_grad
        self.Jb = np.sum(output_grad, axis=0, keepdims=True)

    def get_input_grad(self, Y, output_grad=None, T=None):
        """Return the gradient at the input of this layer"""
        return output_grad @ self.weights.T


class SoftmaxOutputLayer(Layer):
    """The sigmoid layer applies the sigmoid function to its input"""
    def __init__(self, weight_decay=0) -> None:
        super().__init__()
        self.weight_decay = weight_decay
        pass

    def get_output(self, X):
        """Return the output of this layer"""
        return Softmax().call(X)
    
    def get_input_grad(self, Y, T):
        """Return the gradient at the input of this layer"""
        if self.weight_decay > 0:
            return (Y - T) / Y.shape[0] + (self.weight_decay * Y) / Y.shape[0]
        return (Y - T) / Y.shape[0]
    
    def get_cost(self, Y, T, loss="cross_entropy"):
        """Return the cost at the output of this layer"""
        if loss == "mean_squared_error":
            l = MSE()
        else:
            l = CrossEntropy()
        return l.calc_loss(Y, T)
    
   