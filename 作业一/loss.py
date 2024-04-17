from __init__ import *

def get_loss_func(loss: str) -> Callable:
    loss = loss.lower()
    if loss in ['ce', 'cross_entrophy']:
        return cross_entrophy
    elif loss in ['softmax', 'softmax_loss']:
        return softmax_loss
    else:
        raise ValueError("Unknown loss function: {}, please choose one from ['cross_entrophy', 'softmax', 'svm']".format(loss))

def cross_entrophy(x: np.array, y: np.array):
    """
    Computes the categorical cross-entropy loss and gradient.
    Inputs:
    - x: Input data, of shape (N, C), where x[i, j] is the predicted
      probability for the jth class for the ith input.
    - y: Vector of labels, of shape (N,), where y[i] is the true class
      label for x[i] and 0 <= y[i] < C.
    Returns a tuple of:
    - loss: Scalar giving the loss.
    - dx: Gradient of the loss with respect to x.
    """
    N = x.shape[0]
    epsilon = 1e-6
    log_probs = -np.log(np.clip(x[np.arange(N), y], a_min=epsilon, a_max=1-epsilon)) # clip the array to avoid overflow / underflow
    loss = np.mean(log_probs)
    dx = np.copy(x)  
    dx[np.arange(N), y] -= 1  
    dx /= N  
    
    return loss, dx

def softmax_loss(x: np.array, y: np.array):
    """
    Computes the loss and gradient for softmax classification.
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for
      the jth class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label
      for x[i] and 0 <= y[i] < C
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    # use shifted_logits to avoid underflow / overflow
    shifted_logits = x - np.max(x, axis=1, keepdims=True)    
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)

    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = (-1.0 / N) * np.sum(log_probs[np.arange(N), y])
    dx = np.copy(probs)
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx

