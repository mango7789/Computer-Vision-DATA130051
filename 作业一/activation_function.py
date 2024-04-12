from numpy.core.multiarray import array as array
from __init__ import *

class ActivationFunction(ABC):
    @classmethod
    def __new__(cls, type: str):
        if type.lower() == 'relu':
            return ReLU()
        elif type.lower() == 'tanh':
            return Tanh()
        elif type.lower() == 'sigmoid':
            return Sigmoid()
        else:
            raise ValueError("Unknown activation function type: {}, please choose from ['relu', 'tanh', 'sigmoid']".format(type))

    @abstractmethod
    def forward(self, x: np.array):
        raise NotImplementedError
    
    @abstractmethod
    def backward(self, grad: np.array, score: np.array):
        raise NotImplementedError
    
#######################################################################
#                             ReLU                                    #
#######################################################################
class ReLU(ActivationFunction):
    def forward(self, x: np.array):
        return np.max(0, x)
    
    def backward(self, grad: np.array, score: np.array):
        grad[score == 0] = 0
        return grad

#######################################################################
#                             Tanh                                    #
#######################################################################
class Tanh(ActivationFunction):
    def forward(self, x: np.array):
        return np.tanh(x)

    def backward(self, grad: np.array, score: np.array):
        return grad * (1 - score**2)

#######################################################################
#                            Sigmoid                                  #
#######################################################################
class Sigmoid(ActivationFunction):
    def forward(self, x: np.array):
        return 1 / (1 + np.exp(-x))
    
    def backward(self, grad: np.array, score: np.array):
        return grad * score * (1 - score)