import numpy as np

def compute_loss(Y_true, Y_pred):
    """
    Computes the categorical cross-entropy loss.
    Y_true: one-hot encoded true labels (m, n_classes)
    Y_pred: predicted probabilities from softmax (m, n_classes)
    """
    m = Y_true.shape[1]
    # Small epsilon to avoid log(0)
    loss = -np.sum(Y_true * np.log(Y_pred + 1e-8)) / m
    return loss

def compute_accuracy(Y_true, Y_pred):
    """
    Computes the accuracy.
    Y_true: one-hot encoded true labels
    Y_pred: raw predictions from the model (before converting to class index)
    """
    y_true_labels = np.argmax(Y_true, axis=0)
    y_pred_labels = np.argmax(Y_pred, axis=0)
    accuracy = np.mean(y_true_labels == y_pred_labels)
    return accuracy