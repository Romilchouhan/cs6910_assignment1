import numpy as np 
from activations import Softmax

class MSE():
    def __init__(self) -> None:
        pass

    def calc_loss(self, a, y):
        #convert nan to 0
        a = np.nan_to_num(a)

        # check if y is empty
        try:
            y.shape[0]
        except:
            return 0
        return np.sum((a-y)**2)/ (2 * y.shape[0]) # y.shape[0] is the number of samples
    
    def deriv(self, a, y):
        return (a-y)/y.shape[0]

class CrossEntropy():
    def __init__(self) -> None:
        pass

    def calc_loss(self, a, y):
        # print(a)
        epsilon = 1e-6
        self.a = a 
        self.y = y
        loss = -np.sum(np.sum(self.y * np.log(self.a + epsilon))) / self.y.shape[0]
        return loss
    
    def deriv(self, a, y):
        m = y.shape[0]
        grad = Softmax().call(a)
        grad[range(m), y] -= 1
        grad = grad/m
        return grad
    