from __init__ import *

class Activation:
    @classmethod
    def __new__(cls, type: str):
        """
        Create a new activation-component based on the input of type. 
        Inputs:
        - type: the type of activation function, should be in ['relu', 'tanh', 'sigmoid](ignore case).
        """
        type = type.lower()
        if type == 'relu':
            return ReLU()
        elif type == 'tanh':
            return Tanh()
        elif type == 'sigmoid':
            return Sigmoid()
        else:
            raise ValueError("Unknown activation function type: {}, please choose from ['relu', 'tanh', 'sigmoid']".format(type))

    @classmethod
    def forward(cls, x: np.array, type: str):
        """
        Computes the forward pass for an activation component.
        Inputs:
        - x: An array containing input data, of shape (N, M)
         Returns a tuple of:
        - out: output, of shape (N, M)
        - cache: x, used for backward pass
        """
        activation_func = cls(type)
        return activation_func.forward(x)

    @classmethod
    def backward(cls, dout: np.array, cache: np.array, type: str):
        """
        Computes the backward pass for an activation component.
        Inputs:
        - dout: Upstream derivative, of shape (N, M)
        - cache: Tuple of:
          - x: Input data, of shape (N, M)
        Returns a tuple of:
        - dx: Gradient with respect to x, of shape (N, M)
        """
        activation_func = cls(type)
        return activation_func.backward(dout, cache)
    
#######################################################################
#                             ReLU                                    #
#######################################################################
class ReLU(Activation):
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
class Tanh(Activation):
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
class Sigmoid(Activation):
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