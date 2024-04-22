from __init__ import *
from full_connect_network import FullConnectNet
from optimization import get_optim_func

class Solver:
    """
    A Solver encapsulates all the logic necessary for training classification
    models. The Solver performs stochastic gradient descent using different
    update rules.
    The solver accepts both training and validation data and labels so it can
    periodically check classification accuracy on both training and validation
    data to watch out for overfitting.
    To train a model, you will first construct a Solver instance, passing the
    model, dataset, and various options (learning rate, batch size, etc) to the
    constructor. You will then call the train() method to run the optimization
    procedure and train the model.
    After the train() method returns, model.params will contain the parameters
    that performed best on the validation set over the course of training.
    In addition, the instance variable solver.loss_hist will contain a list
    of all losses encountered during training and the instance variables
    solver.train_acc_hist and solver.val_acc_hist will be lists of the
    accuracies of the model on the training and validation set at each epoch.
    Example usage might look something like this:
    data = {
        'X_train': # training data
        'y_train': # training labels
        'X_val': # validation data
        'y_val': # validation labels
    }
    model = FullConnectNet(hidden_size=100, reg=10)
    solver = Solver(
            model, 
            data,
            update_rule='sgd',
            optim_config={
                'learning_rate': 1e-3,
            },
            lr_decay=0.9,
            num_epochs=10, 
            batch_size=100,
            print_iter=100,
        )
    solver.train()
    """
    def __init__(self, model: FullConnectNet, data: Dict, **kwargs) -> None:
        """
        Construct a new Solver instance.
        Required arguments:
        - model: A model object conforming to the API described above
        - data: A dictionary of training and validation data containing:
          'X_train': Array, shape (N_train, d_1, ..., d_k) of training images
          'X_val': Array, shape (N_val, d_1, ..., d_k) of validation images
          'y_train': Array, shape (N_train,) of labels for training images
          'y_val': Array, shape (N_val,) of labels for validation images
        Optional arguments:
        - batch_size: Size of minibatches used to compute loss and gradient
          during training.
        - update_rule: The string format of a function of an update rule. Default is 'sgd'.
        - optim_config: A dictionary containing hyperparameters that will be
          passed to the chosen update rule. Each update rule requires different
          hyperparameters but all update rules require a
          'learning_rate' parameter so that should always be present.
        - lr_decay: A scalar for learning rate decay; after each epoch the
          learning rate is multiplied by this value.
        - num_epochs: The number of epochs to run for during training.
        - iters: The number of iterations to run for the entire training.
        - print_iter: Integer; training losses will be printed every
          print_iter iterations.
        - verbose: Boolean; if set to false then no output will be printed
          during training.
        """
        # get the model and data
        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']

        # unpack arguments related to optimization step
        self.batch_size = kwargs.pop('batch_size', 100)
        self.update_rule = kwargs.pop("update_rule", 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.iters = kwargs.pop('iters', 6000)

        # unpack arguments related to output of the solver
        self.print_iter = kwargs.pop('print_iter', 10)
        self.verbose = kwargs.pop('verbose', True)

        # if there have other undesired arguments
        if len(kwargs) > 0:
            arguments = ", ".join('"%s"' % key for key in list(kwargs.keys()))
            raise ValueError('Unrecognized arguments %s' % arguments)

        self.__reset()

    def __reset(self):
        """
        Initialize some variables for recording the training process.
        """
        self.epoch = 0
        self.loss_hist = []
        self.train_acc_hist = []
        self.val_acc_hist = []
        self.best_val_acc = 0
        self.best_params = {}
        self.optim_configs = {}
        for para in self.model.params:
            if para[0] == 'A':
                continue
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[para] = d

    def step(self):
        """
        Make a single optimization step over a small batch of the training data.
        """
        # sample a small batch of the training data
        N = self.X_train.shape[0]
        batch_mask = np.random.choice(np.arange(N), self.batch_size, replace=False)
        X_batch, y_batch = self.X_train[batch_mask], self.y_train[batch_mask]

        # compute the loss and gradients
        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_hist.append(float(loss))

        # update
        for p, w in self.model.params.items():
            # skip the activation layer and loss function
            if 'A' in p:
                continue
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = get_optim_func(self.update_rule)(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

    def check_accuracy(self, X: np.array, y: np.array, batch_size: int=100):
        """
        Check accuracy of the model on the provided data.
        Inputs:
        - X: Array of data, of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,).
        - batch_size: Split X and y into batches of this size to avoid using
          too much memory.
        Returns:
        - acc: Scalar giving the fraction of instances that were correctly
          classified by the model.
        """
        N = X.shape[0]
        num_batches = N // batch_size
        batch_size = batch_size + 1 if N % batch_size != 0 else batch_size

        y_pred = []
        for i in range(num_batches):
            X_batch = X[i * batch_size: (i + 1) * batch_size]
            scores = self.model.loss(X_batch)
            y_pred.append(np.argmax(scores, axis=1))

        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)

        return float(acc)

    
    def train(self):
        """
        Train the model based on the configuration in the Solver
        """
        iter_per_epoch = max(self.iters // self.num_epochs, 1)
        num_iter = self.iters
        start_time = time.time()

        for t in range(num_iter):

            self.step()

            # print the training loss
            if self.verbose and t % self.print_iter == 0:
                print("(Time {:>6.2f} s; Iteration {:>5} / {:>5}) loss {:>7.6f}".format(
                    time.time() - start_time,
                    t + 1,
                    num_iter, 
                    self.loss_hist[-1]
                ))

            # learning rate decay after one epoch
            epoch_end = (t + 1) % iter_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]['learning_rate'] *= self.lr_decay

            # check train and val accuracy on the first iteration, the last
            # iteration, and at the end of each epoch.
            first_iter = t == 0
            last_iter = t == num_iter - 1
            if first_iter or last_iter or epoch_end:
                train_acc =  self.check_accuracy(self.X_train, self.y_train)
                val_acc = self.check_accuracy(self.X_val, self.y_val)
                self.train_acc_hist.append(train_acc)
                self.val_acc_hist.append(val_acc)

                if self.verbose:
                    print("[Epoch {:>2} / {:>2}] train accuracy: {:>7.6f}; val accuracy: {:>7.6f}".format(
                        self.epoch, self.num_epochs, train_acc, val_acc
                    ))

                # keep track of the best model
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        if 'A' in k:
                            self.best_params[k] = v
                        else:
                            self.best_params[k] = np.copy(v)

            # replace the parameters with the best parameters
            self.model.params = self.best_params

    def save(self, path: str):
        """
        Save the best model
        """
        self.model.save(path)

    def load(self, path: str, dtype: np.dtype=np.float64):
        """
        Load the pre-trained model in the given path
        """
        self.model.load(path, dtype)