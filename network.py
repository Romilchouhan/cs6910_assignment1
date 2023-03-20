import numpy as np 
from activations import Sigmoid, Softmax, ReLU, Tanh, Identity, LeakyReLU, ELU
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
    def __init__(self, activation='sigmoid', mode='train') -> None:
        super().__init__()
        self.mode = mode
        if activation == 'Identity':
            self.activation = Identity()
        elif activation == 'ReLU':
            self.activation = ReLU()
        elif activation == 'tanh':
            self.activation = Tanh()
        elif activation == 'LeakyReLU':
            self.activation = LeakyReLU()
        elif activation == 'elu':
            self.activation = ELU()
        else:  # default activation
            self.activation = Sigmoid()
    
    def get_output(self, X, mode='train'):
        """Return the output of this layer"""
        return self.activation.call(X)

    def get_input_grad(self, Y, output_grad=None, T=None):
        """Return the gradient of the loss with respect to the input of this layer"""
        return np.multiply(output_grad, self.activation.deriv(Y)) / Y.shape[0]



class LinearLayer(Layer):
    """The linear layer performs a linear transformation to its input"""
    def __init__(self, in_features, out_features, initialisation, mode='train') -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.initialisation = initialisation 
        self.mode = mode
        self.weight_initialisation()
        if self.initialisation == 'random':
            self.bias = np.random.normal(loc=0, scale=1.0, size=out_features) * 0.1
        else:
            self.bias = np.zeros(out_features)
        self.ut_w = 0  # momentum gradient descent
        self.ut_b = 0  # momentum gradient descent
        self.mt_w = 0  # RMSprop, adam and nadam
        self.mt_b = 0 # RMSprop, adam and nadam
        self.vt_w = 0  # adam and nadam
        self.vt_b = 0  # adam and nadam
    
    def weight_initialisation(self):
        if (self.initialisation == 'random'):
            self.weights = np.random.randn(self.in_features, self.out_features) * 0.1

        elif (self.initialisation == 'xavier'):
            upper_bound = np.sqrt(6.0 / (self.in_features + self.out_features))
            lower_bound = -np.sqrt(6.0 / (self.in_features + self.out_features))
            self.weights = np.random.uniform(lower_bound, upper_bound, (self.in_features, self.out_features))

        elif (self.initialisation == 'he'):
            self.weights = np.random.randn(self.in_features, self.out_features) * np.sqrt(2 / self.in_features)
    
    def get_output(self, X, mode='train'):
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
    def __init__(self, weight_decay=0, mode='train') -> None:
        super().__init__()
        self.weight_decay = weight_decay
        self.mode = mode
        pass

    def get_output(self, X, mode='train'):
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
    
   
class DropoutLayer(Layer):
    """The dropout layer applies the dropout function to its input"""
    def __init__(self, dropout_rate, mode='train') -> None:
        super().__init__()
        self.dropout_rate = dropout_rate
        self.mode = mode
        pass 

    def get_output(self, X, mode='train'):
        """Return the output of this layer"""
        if mode == 'train':
            self.keep_prob = 1 - self.dropout_rate
            self.D1 = np.random.rand(X.shape[0], X.shape[1])
            self.D1 = self.D1 < self.keep_prob
            self.D1 = self.D1.astype(int)
            X = np.multiply(X, self.D1)
            X = X / self.dropout_rate
        return X
    
    def get_input_grad(self, Y, output_grad=None, T=None):
        """Return the gradient at the input of this layer"""
        dA = np.multiply(output_grad, self.D1) / Y.shape[0]
        return dA / self.keep_prob
    

class BatchNormLayer(Layer):
    """The batch normalization layer applies the batch normalization to the batches of its input"""
    def __init__(self, gamma, beta, epsilon=1e-5, momentum=0.9) -> None:
        super().__init__()
        self.gamma = gamma
        self.beta = beta
        self.epsilon = epsilon
        self.momentum = momentum
        self.running_mean = 0
        self.running_var = 0
        pass

    def get_output(self, X, mode='train'):    
        """Return the output of this layer"""
        if mode == 'train':
            self.mean = np.mean(X, axis=0)
            self.var = np.var(X, axis=0)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var
        
        elif mode == 'test':
            self.mean = self.running_mean
            self.var = self.running_var

        self.std = np.sqrt(self.var + self.epsilon)
        self.X_norm = (X - self.mean) / np.sqrt(self.var + self.epsilon)
        self.out = self.gamma * self.X_norm + self.beta
        return self.out

    def get_input_grad(self, Y, output_grad=None, T=None):
        """Return the gradient at the input of this layer"""
        m = Y.shape[0]
        self.dgamma = np.sum(output_grad * self.X_norm, axis=0)
        self.dbeta = np.sum(output_grad, axis=0)
        dX_norm = output_grad * self.gamma
        dX = (1. / m) * self.std * (m * dX_norm - np.sum(dX_norm, axis=0) - self.X_norm * np.sum(dX_norm * self.X_norm, axis=0))
        return dX