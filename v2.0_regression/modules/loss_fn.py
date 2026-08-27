import numpy as np

# Mean Squared Error
def mse_loss(y_pred, y_true):
    """
    Mean Squared Error (MSE) loss.
    """
    return np.mean((y_pred - y_true) ** 2)

def mse_deriv(y_pred, y_true, z, act_deriv):
    """
    Derivative of MSE loss w.r.t. pre-activation z.
    Requires activation function derivative.
    """
    return 2 * (y_pred - y_true) * act_deriv(z)

# Cross Entropy
def cross_entropy_loss(y_pred, y_true, eps=1e-12):
    """
    Cross-entropy loss.
    Assumes y_true is one-hot. Applies clipping to avoid log(0).
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[1]

def cross_entropy_deriv(y_pred, y_true, z=None, act_deriv=None):
    """
    Derivative of cross-entropy loss.
    Assumes softmax was applied in final layer.
    Simplifies to y_pred - y_true.
    """
    return y_pred - y_true

# Loss function getter
def get_lossfn(name: str):
    """
    Retrieve a loss function and its derivative by name prefix.

    Parameters:
        name (str): Base name of the loss function (e.g., 'mse', 'cross_entropy')

    Returns:
        tuple: (activation_function, activation_derivative)

    Raises:
        ValueError: If the name is not recognized.
    """
    loss_fns = {
        "mse": (mse_loss, mse_deriv),
        "cross_entropy": (cross_entropy_loss, cross_entropy_deriv),
    }
    try:
        return loss_fns[name]
    except KeyError:
        raise ValueError(f"Loss function '{name}' not defined. Available: {list(loss_fns)}")
