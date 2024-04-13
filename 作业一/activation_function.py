from __init__ import *

class ActivationFunction:
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

    @classmethod
    def forward(cls, x: np.array, type: str):
        activation_func = cls(type)
        return activation_func.forward(x)

    @classmethod
    def backward(cls, dout: np.array, cache: np.array, type: str):
        activation_func = cls(type)
        return activation_func.backward(dout, cache)
    
#######################################################################
#                             ReLU                                    #
#######################################################################
class ReLU(ActivationFunction):
    def forward(self, x: np.array):
        out = np.copy(x)
        out[out < 0] = 0
        cache = x
        return out, cache
    
    def backward(self, dout: np.array, cache: np.array):
        dx, x = None, cache
        dx = np.copy(dout)
        dx[x <= 0] = 0
        return dx

#######################################################################
#                             Tanh                                    #
#######################################################################
class Tanh(ActivationFunction):
    def forward(self, x: np.array):
        out = np.tanh(np.copy(x))
        cache = x
        return out, cache

    def backward(self, dout: np.array, cache: np.array):
        dx, x = None, cache
        dx = dout * (1 - np.power(np.tanh(x), 2))
        return dx

#######################################################################
#                            Sigmoid                                  #
#######################################################################
class Sigmoid(ActivationFunction):
    def sigmoid(self, x: np.array):
        return 1 / (1 + np.exp(-x))

    def forward(self, x: np.array):
        out = np.copy(x)
        out = self.sigmoid(out)
        cache = x
        return out, cache
    
    def backward(self, dout: np.array, cache: np.array):
        dx, x = None, cache
        dx = dout * self.sigmoid(x) * (1 - self.sigmoid(x))
        return dx