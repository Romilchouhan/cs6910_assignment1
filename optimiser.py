import numpy as np 
from network import Layer, LinearLayer, SoftmaxOutputLayer, ActivationLayer

# LET'S SAY YOU HAVE THE BACKWARD GRADIENTS
class SGD(object):
    """Stochastic gradient descent optimizer"""
    def __init__(self, learning_rate) -> None:
        super().__init__()
        self.lr = learning_rate

    def update_params(self, layers):
        for layer in layers:
            if isinstance(layer, LinearLayer):
                layer.weights -= self.lr * layer.JW   
                layer.bias -= self.lr * layer.Jb.reshape(layer.bias.shape)

class MGD(object):
    """Momentum gradient descent optimizer"""
    def __init__(self, learning_rate, momentum = 0.9) -> None:
        super().__init__()
        self.lr = learning_rate
        self.beta = momentum

    def update_params(self, layers):
        for layer in layers: 
            if isinstance(layer, LinearLayer): 
                 layer.ut_w = self.beta * layer.ut_w + layer.JW
                 layer.weights -= self.lr * layer.ut_w

                 layer.ut_b = self.beta * layer.ut_b + layer.Jb.reshape(layer.bias.shape)
                 layer.bias -= self.lr * layer.ut_b

class NAG(object):
    """Nesterov accelerated gradient descent optimizer"""
    def __init__(self, learning_rate, momentum = 0.9) -> None:
        super().__init__()
        self.lr = learning_rate
        self.beta = momentum

    def update_params(self, layers):
        for layer in layers: 
            if isinstance(layer, LinearLayer): 
                layer.ut_w = self.beta * layer.ut_w + self.lr * layer.JW
                layer.weights -= self.lr * (self.beta * layer.ut_w + layer.JW)

                layer.ut_b = self.beta * layer.ut_b + self.lr * layer.Jb.reshape(layer.bias.shape)
                layer.bias -= self.lr * (self.beta * layer.ut_b + layer.Jb.reshape(layer.bias.shape))

class AdaGrad(object):
    """AdaGrad adaptive optimization algorithm"""
    def __init__(self, lr, epsilon = 1e-6) -> None:
        super().__init__()
        self.lr = lr
        self.epsilon = epsilon

    def update_params(self, layers):
        for layer in layers:
            if isinstance(layer, LinearLayer):
                layer.ut_w = layer.ut_w + np.square(layer.JW)
                layer.weights -= (self.lr / np.sqrt(layer.ut_w + self.epsilon)) * layer.JW

                layer.ut_b = layer.ut_b + np.square(layer.Jb.reshape(layer.bias.shape))
                layer.bias -= (self.lr / np.sqrt(layer.ut_b + self.epsilon)) * layer.Jb.reshape(layer.bias.shape)
                layer.bias -= (self.lr / np.sqrt(layer.ut_b + self.epsilon)) * layer.Jb.reshape(layer.bias.shape)


class RMSProp(object):
    """RMSprop adaptive optimization algorithm"""
    def __init__(self, lr, beta = 0.9, epsilon = 1e-6) -> None:
        super().__init__()
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon

    def update_params(self, layers):
        for layer in layers:
            if isinstance(layer, LinearLayer):
                layer.ut_w = self.beta * layer.ut_w + (1 - self.beta) * np.square(layer.JW)
                layer.weights -= (self.lr / np.sqrt(layer.ut_w + self.epsilon)) * layer.JW

                layer.ut_b = self.beta * layer.ut_b + (1 - self.beta) * np.square(layer.Jb.reshape(layer.bias.shape))
                layer.bias -= (self.lr / np.sqrt(layer.ut_b + self.epsilon)) * layer.Jb.reshape(layer.bias.shape)

class Adam(object):
    """Adam adaptive optimization algorithm"""
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8) -> None:
        super().__init__()
        self.m = 0  # moving average of the gradient
        self.v = 0  # moving average of the squared gradient
        self.t = 1  # time step
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def update_params(self, layers):
        for layer in layers:
            if isinstance(layer, LinearLayer):
                layer.mt_w = self.beta1 * layer.mt_w + (1-self.beta1) * layer.JW
                layer.vt_w = self.beta2 * layer.vt_w + (1-self.beta2) * np.square(layer.JW)
                mt_hat_w = layer.mt_w / (1-self.beta1**self.t)
                vt_hat_w = layer.vt_w / (1-self.beta2**self.t)

                layer.mt_b = self.beta1 * layer.mt_b + (1-self.beta1) * layer.Jb.reshape(layer.bias.shape)
                layer.vt_b = self.beta2 * layer.vt_b + (1-self.beta2) * np.square(layer.Jb.reshape(layer.bias.shape))
                mt_hat_b = layer.mt_b / (1-self.beta1**self.t)
                vt_hat_b = layer.vt_b / (1-self.beta2**self.t)

                layer.weights -= self.lr * mt_hat_w / (np.sqrt(vt_hat_w) + self.epsilon)
                layer.bias -= self.lr * mt_hat_b / (np.sqrt(vt_hat_b) + self.epsilon)
        self.t += 1


class Nadam(object):
    """Nestorov adam adaptive optimization algorithm"""
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8) -> None:
        super().__init__()
        self.m = 0  # moving average of the gradient
        self.v = 0  # moving average of the squared gradient
        self.t = 1  # time step
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def update_params(self, layers):
        for layer in layers:
            if isinstance(layer, LinearLayer):
                layer.mt_w = self.beta1 * layer.mt_w + (1-self.beta1) * layer.JW
                layer.vt_w = self.beta2 * layer.vt_w + (1-self.beta2) * np.square(layer.JW)
                mt_hat_w = layer.mt_w / (1-self.beta1**self.t)
                vt_hat_w = layer.vt_w / (1-self.beta2**self.t)

                layer.mt_b = self.beta1 * layer.mt_b + (1-self.beta1) * layer.Jb.reshape(layer.bias.shape)
                layer.vt_b = self.beta2 * layer.vt_b + (1-self.beta2) * np.square(layer.Jb.reshape(layer.bias.shape))
                mt_hat_b = layer.mt_b / (1-self.beta1**self.t)
                vt_hat_b = layer.vt_b / (1-self.beta2**self.t)

                layer.weights -= self.lr * mt_hat_w / (np.sqrt(vt_hat_w) + self.epsilon)
                layer.bias -= self.lr * mt_hat_b / (np.sqrt(vt_hat_b) + self.epsilon)
        self.t += 1
