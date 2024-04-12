from __init__ import *

def nn_forward_pass(
    params: Dict[str, np.array | str], 
    X: np.array
):
    pass

def nn_backward_pass(
    params: Dict[str, np.array | str], 
    grads: Dict[str, np.array | str], 
    X: np.array, 
    y: np.array
):
    pass

def nn_forward_backward(
    params: Dict[str, np.array | str],
    X: np.array,
    y: Optional[np.array]=None,
    reg: float=0.0        
):
    pass

def nn_train(
    params: Dict[str, np.array | str],
    loss_func: Callable,
    pred_func: Callable,
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
    pass

def nn_predict(
    params: Dict[str, np.array | str], 
    loss_func: Callable, 
    X: np.array
):
    pass


def nn_get_search_params():
    pass

def find_best_net(
    data_dict: Dict[str, np.array | str], 
    get_param_set_fn: Callable
):
    pass