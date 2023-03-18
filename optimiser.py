import numpy as np 

# LET'S SAY YOU HAVE THE BACKWARD GRADIENTS
class SGD(object):
    """Stochastic gradient descent optimizer"""
    def __init__(self, learning_rate) -> None:
        super().__init__()
        self.lr = learning_rate

    def update_params(self, layers):
        for linear, _ in layers:
            linear.weights -= self.lr * linear.JW   
            linear.bias -= self.lr * linear.Jb.reshape(linear.bias.shape)
    

class MGD(object):
    """Momentum gradient descent optimizer"""
    def __init__(self, learning_rate, momentum = 0.9) -> None:
        super().__init__()
        self.lr = learning_rate
        self.beta = momentum

    def update_params(self, layers):
        for linear, _ in layers:
            linear.ut_w = self.beta * linear.ut_w + linear.JW
            linear.weights -= self.lr * linear.ut_w

            linear.ut_b = self.beta * linear.ut_b + linear.Jb.reshape(linear.bias.shape)
            linear.bias -= self.lr * linear.ut_b       

class NAG(object):
    """Nesterov accelerated gradient descent optimizer"""
    def __init__(self, learning_rate, momentum = 0.9) -> None:
        super().__init__()
        self.lr = learning_rate
        self.beta = momentum
    
    def update_params(self, layers):
        for linear, _ in layers: 
            linear.ut_w = self.beta * linear.ut_w + self.lr * linear.JW
            # linear.u_t = self.beta * linear.u_t + linear.JW
            # linear.weights -= self.lr * linear.u_t
            linear.weights -= self.lr * (self.beta * linear.ut_w + linear.JW)

            linear.ut_b = self.beta * linear.ut_b + self.lr * linear.Jb.reshape(linear.bias.shape)
            linear.bias -= self.lr * (self.beta * linear.ut_b + linear.Jb.reshape(linear.bias.shape))

class AdaGrad(object):
    """AdaGrad adaptive optimization algorithm"""
    def __init__(self, lr, epsilon = 1e-6) -> None:
        super().__init__()
        self.lr = lr
        self.epsilon = epsilon

    def update_params(self, layers):
        for linear, _ in layers:
            linear.ut_w = linear.ut_W + np.square(linear.JW)
            linear.weights -= (self.lr / np.sqrt(linear.ut_w + self.epsilon)) * linear.JW

            linear.ut_b = linear.ut_b + np.square(linear.Jb.reshape(linear.bias.shape))
            linear.bias -= (self.lr / np.sqrt(linear.ut_b + self.epsilon)) * linear.Jb.reshape(linear.bias.shape)
            linear.bias -= (self.lr / np.sqrt(linear.ut_b + self.epsilon)) * linear.Jb.reshape(linear.bias.shape)


## THIS HAS SOME PROBLEM IN CONVERGENCE
class RMSProp(object):
    """RMSprop adaptive optimization algorithm"""
    def __init__(self, lr, beta = 0.9, epsilon = 1e-6) -> None:
        super().__init__()
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
    
    def update_params(self, layers):
        for linear, _ in layers:
            linear.ut_w = self.beta * linear.ut_w + (1 - self.beta) * np.square(linear.JW)
            linear.weights -= (self.lr / np.sqrt(linear.ut_w + self.epsilon)) * linear.JW

            linear.ut_b = self.beta * linear.ut_b + (1 - self.beta) * np.square(linear.Jb.reshape(linear.bias.shape))
            linear.bias -= (self.lr / np.sqrt(linear.ut_b + self.epsilon)) * linear.Jb.reshape(linear.bias.shape)

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

        # layer.b -= self.lr*m_hat_b/(np.sqrt(v_hat_b)+self.eps)


    def update_params(self, layers):
        for linear, _ in layers:
            linear.mt_w = self.beta1 * linear.mt_w + (1-self.beta1) * linear.JW
            linear.vt_w = self.beta2 * linear.vt_w + (1-self.beta2) * np.square(linear.JW)
            mt_hat_w = linear.mt_w / (1-self.beta1**self.t)
            vt_hat_w = linear.vt_w / (1-self.beta2**self.t)

            linear.mt_b = self.beta1 * linear.mt_b + (1-self.beta1) * linear.Jb.reshape(linear.bias.shape)
            linear.vt_b = self.beta2 * linear.vt_b + (1-self.beta2) * np.square(linear.Jb.reshape(linear.bias.shape))
            mt_hat_b = linear.mt_b / (1-self.beta1**self.t)
            vt_hat_b = linear.vt_b / (1-self.beta2**self.t)

            linear.weights -= self.lr * mt_hat_w / (np.sqrt(vt_hat_w) + self.epsilon)
            linear.bias -= self.lr * mt_hat_b / (np.sqrt(vt_hat_b) + self.epsilon)
        self.t += 1


## THIS HAS SOME PROBLEM IN CONVERGENCE
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
        for linear, _ in layers:
            linear.mt_w = self.beta1 * linear.mt_w + (1-self.beta1) * linear.JW
            linear.vt_w = self.beta2 * linear.vt_w + (1-self.beta2) * np.square(linear.JW)
            mt_hat_w = linear.mt_w / (1-self.beta1**self.t)
            vt_hat_w = linear.vt_w / (1-self.beta2**self.t)
            linear.weights -= self.lr * mt_hat_w / (np.sqrt(vt_hat_w) + self.epsilon)

            linear.mt_b = self.beta1 * linear.mt_b + (1-self.beta1) * linear.Jb.reshape(linear.bias.shape)
            linear.vt_b = self.beta2 * linear.vt_b + (1-self.beta2) * np.square(linear.Jb.reshape(linear.bias.shape))
            mt_hat_b = linear.mt_b / (1-self.beta1**self.t)
            vt_hat_b = linear.vt_b / (1-self.beta2**self.t)
            linear.bias -= self.lr * mt_hat_b / (np.sqrt(vt_hat_b) + self.epsilon)
        self.t += 1
