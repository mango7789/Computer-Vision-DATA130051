from __init__ import *
from linear_layer import Linear
from loss import Loss

class FullConnectNet:

    def __init__(
            self,
            hidden_dims: List[int],
            types: List[Literal['relu', 'tanh', 'sigmoid']],
            input_dim: int=3*28*28,
            num_classes: int=10,
            reg: float=0.0,
            weight_scale: float=1e-2,
            loss: str='cross_entrophy',
            dtype: np.dtype=np.float64
        ):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each
          hidden layer.
		    - types: A list of strings giving the type of each activation function.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - loss: The loss function to be used.
        - dtype: A numpy data type object; all computations will be
          performed using this datatype.
        """
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        # only use one type of activation function
        if len(type) == 1 and self.num_layers > 2:
            types = types * (self.num_layers - 1)
        # unmatching number of activation functions and hidden layers
        elif len(type) != 1 and len(type) != self.num_layers - 1:
            raise ValueError("The number of activation functions should be 1 or the same as the number of hidden layers minus 1.") 
        
        self.params['W1'] = np.random.randn(input_dim, hidden_dims[0], dtype=dtype) * weight_scale
        self.params['b1'] = np.zeros(hidden_dims[0], dtype=dtype)
        self.params['A1'] = types[0]
        for i in range(2, self.num_layers):
            self.params[f'W{i}'] = np.random.randn(hidden_dims[i-2], hidden_dims[i-1], dtype=dtype) * weight_scale
            self.params[f'b{i}'] = np.zeros(hidden_dims[i-1], dtype=dtype)
            self.params[f'A{i}'] = types[i-1]
        self.params[f'W{self.num_layers}'] = np.random.randn(hidden_dims[self.num_layers-2], num_classes, dtype=dtype) * weight_scale
        self.params[f'b{self.num_layers}'] = np.zeros(num_classes, dtype=dtype)
        self.params['loss'] = loss

    def loss(self, X: np.array, y: np.array=None):
        """
        Compute loss and gradient for the fully-connected net.

        Inputs:
        - X: A numpy array of input data of shape (N, d_1, ..., d_k)
        - y: numpy array of labels, of shape (N,) and type int. y[i] gives the
          label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model
        and return:
        - scores: Array of shape (N, C) giving classification scores,
          where scores[i, c] is the classification score for X[i]
          and class c.
        If y is not None, then run a training-time forward and backward
        pass and return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping
          parameter names to gradients of the loss with respect to
          those parameters.
        """
        X = X.astype(self.dtype)
        h = X
        activation_caches = {}
        # forward pass for the full-connected net. Compute the class scores for X and storing them in the scores variable
        for i in range(1, self.num_layers):
            h, activation_caches[i] = Linear_Activation.forward(h, self.params[f'W{i}'], self.params[f'b{i}'], self.params[f'A{i}'])
        scores, cache = Linear.forward(h, self.params[f'W{self.num_layers}'], self.params[f'b{self.num_layers}'])

        loss, grads = 0.0, {}
        loss, dout = self.params['loss'](scores, y)

        # add regularization losses
        for i in range(1, self.num_layers + 1):
            loss += self.reg * np.sum(self.params[f'W{i}'] ** 2)

        # backward pass for the fully-connected net. Store the loss in the loss variable and gradients in the grads dictionary
        dh, grads[f'W{self.num_layers}'], grads[f'b{self.num_layers}'] = Linear.backward(dout, cache)
        grads[f'W{self.num_layers}'] += 2 * self.reg * self.params[f'W{self.num_layers}']
        for j in range(self.num_layers - 1, 0, -1):
            dh, grads[f'W{j}'], grads[f'b{j}'] = Linear_Activation.backward(dh, activation_caches[j])
            grads[f'W{j}'] += 2 * self.reg * self.params[f'W{j}']

        return loss, grads

    def save(self, path: str):
        """
        Save the model parameters in a `.npy` file in the `model` directory.

        Inputs:
        - path: The file name of the zipped model parmeters file.
        """
        save_params = {
            'reg': self.reg,
            'num_layers': self.num_layers,
            'dtype': self.dtype,
            'params': self.params
        }
        os.makedirs('model', exist_ok=True)
        path = os.path.join('model', path)
        np.savez(path, **save_params)
        print("Model has been saved in {}".format(path))

    def load(self, path: str, dtype: np.dtype=np.float64):
        """
        Load the model parameters from a `.npy` file in the `model` directory.

        Inputs:
        - path: The file name of the zipped model parmeters file.
        - dtype: A numpy datatype object; all parameters in the model parameters file will
          be converted to this datatype.
        """
        save_params = np.load(os.path.join('model', path))
        self.reg = save_params['reg']
        self.num_layers = save_params['num_layers']
        self.dtype = save_params['dtype']
        self.params = save_params['params'][()]

        if dtype != self.dtype:
            for p in self.params:
                if p[0] != 'A':
                    self.params[p] = self.params[p].astype(dtype)

        print("Successfully load model file: {}".format(path))
