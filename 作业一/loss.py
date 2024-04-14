from __init__ import *

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
    
    # add epsilon to avoid underflow / overflow during logarithm calculations
    epsilon = 1e-15
    x = np.clip(x, epsilon, 1 - epsilon)
    
    log_probs = -np.log(x[np.arange(N), y])
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
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdim=True)
    log_probs = shifted_logits - np.log(Z)

    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = (-1.0 / N) * np.sum(log_probs[np.arange(N), y])
    dx = np.copy(probs)
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx

def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.clip(x - correct_class_scores[:, None] + 1.0, a_min=0.)
    margins[np.arange(N), y] = 0.
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1.
    dx[np.arange(N), y] -= num_pos.astype(dx.dtype)
    dx /= N
    return loss, dx
