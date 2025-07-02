import numpy as np

def sigmoid(Z):
    """Computes the sigmoid activation."""
    return 1 / (1 + np.exp(-Z))

def sigmoid_prime(Z):
    """Computes the derivative of the sigmoid function."""
    s = sigmoid(Z)
    return s * (1 - s)

def relu(Z):
    """Computes the ReLU activation."""
    return np.maximum(0, Z)

def relu_prime(Z):
    """Computes the derivative of the ReLU function."""
    return np.where(Z > 0, 1, 0)

def softmax(Z):
    """
    Computes the softmax activation for a batch of inputs.
    Subtracting the max value improves numerical stability.
    """
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)