import numpy as np 

class Identity():
    def __init__(self) -> None:
        pass

    def call(self, x):
        return x
    
    def deriv(self, x):
        return 1

class Sigmoid(): 
    def __init__(self) -> None:
        pass

    def call(self, x):
        if x.any() < 0:
            return np.exp(x)/(1+np.exp(x))
        else:
            # clip to avoid overflow
            x = np.clip(x, -500, 500)
            return 1/(1+np.exp(-x))
    
    def deriv(self, x):
        return np.multiply(self.call(x), (1 - self.call(x)))
    
class Tanh():
    def __init__(self) -> None:
        pass

    def call(self, x):
        return np.tanh(x)
    
    def deriv(self, x):
        return 1 - np.tanh(x)**2
    
class ReLU():
    def __init__(self) -> None:
        pass

    def call(self, x):
        # clip to avoid overflow
        # x = np.clip(x, -500, 500)
        return np.maximum(0, x)
    
    def deriv(self, x):
        return np.where(x > 0, 1, 0)

class LeakyReLU():
    def __init__(self, alpha=0.01) -> None:
        self.alpha = alpha

    def call(self, x):
        return np.where(x > 0, x, x * self.alpha)
    
    def deriv(self, x):
        return np.where(x > 0, 1, self.alpha)
    
class ELU():
    def __init__(self, alpha=0.01) -> None:
        self.alpha = alpha

    def call(self, x):
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
    
    def deriv(self, x):
        return np.where(x > 0, 1, self.call(x) + self.alpha)

class Softmax():
    def __init__(self) -> None:
        pass

    def call(self, x):
        exps = np.exp(x - np.max(x))
        if exps.ndim == 1:
            return exps / np.sum(exps, axis=0)
        else:
            return exps / np.sum(exps, axis=1, keepdims=True)
    
    def deriv(self, x):
        return self.call(x) * (1 - (1 * self.call(x)).sum(axis=1, keepdims=True)[:, None])