from __init__ import *
from activation_function import *
from nn_func import *
from utils import *

class ThreeLayerNet(object):
    def __init__(
        self,
        input_size: int,
        hidden_size: List[int],
        activation_func: List[Literal['relu', 'tanh', 'sigmoid']],
        output_size: int,
        dtype: np.dtype = np.float64,
        std: float = 1e-4,
    ):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Activation Function are initialized as 
        the string format of the type. Weights, biases and activation functions
        are stored in the variable self.params, which is a dictionary with the 
        following keys:

        W1: First layer weights; has shape (D, H1)
        b1: First layer biases; has shape (H1,)
        A1: First activation function, string
        W2: Second layer weights; has shape (H1, H2)
        b2: Second layer biases; has shape (H2,)
        A2: Second activation function, string
        W3: Third layer weights; has shape (H2, C)
        b3: Third layer biases; has shape (C,)
        """
        self.__check_input(hidden_size, activation_func)
        random.seed(0)
        
        self.params = {}
        # first linear layer and activation function
        self.params['W1'] = std * np.random.randn(
            input_size, hidden_size[0], dtype=dtype
        )
        self.params['b1'] = np.zeros(hidden_size[0], dtype=dtype)
        self.params['A1'] = activation_func[0]
        # second linear layer and activation function
        self.params['W2'] = std * np.random.randn(
            hidden_size[0], hidden_size[1], dtype=dtype
        )
        self.params['b2'] = np.zeros(hidden_size[1], dtype=dtype)
        self.params['A2'] = activation_func[1]
        # third linear layer
        self.params['W3'] = std * np.random.randn(
            hidden_size[1], output_size, dtype=dtype
        )
        self.params['b3'] = np.zeros(output_size, dtype=dtype)

    def __check_input(self, hidden: List[int], func: List[Literal['relu', 'tanh', 'sigmoid']]) -> None:
        """
        Check if the input of the hidden_size and activation_func type is valid.
        """
        if len(hidden) != len(func):
            raise ValueError("The length of hidden_size must match the length of activation_func.")
        if len(hidden) != 2:
            raise ValueError("The length of hidden_size and activation_func must be 2.")
        return
    
    def train(
        self,
        X: np.array,
        y: np.array,
        X_val: np.array,
        y_val: np.array,
        learning_rate: float = 1e-3,
        learning_rate_decay: float = 0.95,
        reg: float = 5e-6,
        num_iters: int = 100,
        batch_size: int = 200,
        verbose: bool = False,
    ):
        return nn_train(
            self.params, nn_forward_backward, nn_predict, X, y,
            X_val, y_val, learning_rate, learning_rate_decay,
            reg, num_iters, batch_size, verbose,
        )

    def save(self, path: str):
        np.savez(self.params, path)
        print("Model saved in {}".format(path))

    def load(self, path: str):
        checkpoint = np.load(path)
        self.params = checkpoint
        if len(self.params) != 8:
            raise Exception("Failed to load your checkpoint")

        for param in ["W1", "b1", "A1", "W2", "b2", "A2", "W3", "b3"]:
            if param not in self.params:
                raise Exception("Failed to load your checkpoint")

    def predict(self, X: np.array):
        return nn_predict(self.params, nn_forward_backward, X)