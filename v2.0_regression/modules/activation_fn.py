"""
activation_fn.py

This module provides a set of commonly used activation functions and their derivatives
for use in neural networks.

Use `get_actfn(name)` to retrieve a function by name.
"""

import numpy as np

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    """Derivative of the sigmoid function."""
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    """Hyperbolic tangent activation function."""
    return np.tanh(x)

def tanh_deriv(x):
    """Derivative of the tanh function."""
    return 1 - np.tanh(x)**2

def linear(x):
    """Linear activation (identity function)."""
    return x

def linear_deriv(x):
    """Derivative of the linear function (constant 1)."""
    return np.ones_like(x)

def relu(x):
    """ReLU activation function (max(0, x))."""
    return np.maximum(0, x)

def relu_deriv(x):
    """Derivative of the ReLU function."""
    return (x > 0).astype(float)

def softmax(z):
    """
    Softmax activation function for classification.
    Applies softmax across columns (i.e., axis=0) for a batch of column vectors.
    """
    exp_z = np.exp(z - np.max(z))  # Subtract max(z) for numerical stability
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def softmax_deriv(z):
    """
    Placeholder for softmax derivative. Not used explicitly when paired with
    cross-entropy loss, since the gradient simplifies to (y_pred - y_true).
    """
    return None

def get_actfn(name: str):
    """
    Retrieve an activation function and its derivative by name prefix.

    Parameters:
        name (str): Base name of the activation function (e.g., 'sigmoid', 'relu', 'tanh')

    Returns:
        tuple: (activation_function, activation_derivative)

    Raises:
        ValueError: If the name is not recognized.
    """
    fn_pairs = {
        "sigmoid": (sigmoid, sigmoid_deriv),
        "tanh": (tanh, tanh_deriv),
        "linear": (linear, linear_deriv),
        "relu": (relu, relu_deriv),
        "softmax": (softmax, softmax_deriv),
    }
    try:
        return fn_pairs[name]
    except KeyError:
        raise ValueError(f"Activation function '{name}' is not defined. Available: {list(fn_pairs)}")
