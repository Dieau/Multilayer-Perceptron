import numpy as np
from .activations import sigmoid, sigmoid_prime, relu, relu_prime, softmax
from .loss import compute_loss, compute_accuracy
from .utils import Colors, Scaler
import time
import copy

class MLP:
    def __init__(self, layer_dims, activations):
        if len(layer_dims) - 2 != len(activations):
            raise ValueError("The number of activation functions must match the number of hidden layers.")
        
        self.layer_dims = layer_dims
        self.activations = list(activations) + ['softmax']
        self.params = {}
        self.num_layers = len(layer_dims)
        self._initialize_parameters()

    def _initialize_parameters(self):
        for l in range(1, self.num_layers):
            self.params[f'W{l}'] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * np.sqrt(2. / self.layer_dims[l-1])
            self.params[f'b{l}'] = np.zeros((self.layer_dims[l], 1))

    def _get_activation_function(self, name):
        if name == 'sigmoid': return sigmoid
        if name == 'relu': return relu
        if name == 'softmax': return softmax
        raise ValueError(f"Unknown activation function: {name}")

    def _get_activation_prime(self, name):
        if name == 'sigmoid': return sigmoid_prime
        if name == 'relu': return relu_prime
        raise ValueError(f"No prime function for: {name}")

    def forward(self, X):
        cache = {}
        A = X
        cache['A0'] = X
        for l in range(1, self.num_layers):
            A_prev = A
            W, b = self.params[f'W{l}'], self.params[f'b{l}']
            Z = np.dot(W, A_prev) + b
            A = self._get_activation_function(self.activations[l-1])(Z)
            cache[f'Z{l}'], cache[f'A{l}'] = Z, A
        return A, cache

    def backward(self, Y_true, cache):
        grads = {}
        m = Y_true.shape[1]
        A_final = cache[f'A{self.num_layers-1}']
        dZ = A_final - Y_true
        for l in reversed(range(1, self.num_layers)):
            A_prev = cache[f'A{l-1}']
            W = self.params[f'W{l}']
            grads[f'dW{l}'] = (1/m) * np.dot(dZ, A_prev.T)
            grads[f'db{l}'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            if l > 1:
                dA_prev = np.dot(W.T, dZ)
                Z_prev = cache[f'Z{l-1}']
                dZ = dA_prev * self._get_activation_prime(self.activations[l-2])(Z_prev)
        return grads

    def update_params(self, grads, learning_rate):
        for l in range(1, self.num_layers):
            self.params[f'W{l}'] -= learning_rate * grads[f'dW{l}']
            self.params[f'b{l}'] -= learning_rate * grads[f'db{l}']

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=150, learning_rate=0.01, batch_size=32, patience=15, verbose=True):
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_params = copy.deepcopy(self.params)
        best_epoch = 0

        for epoch in range(1, epochs + 1):
            permutation = np.random.permutation(X_train.shape[1])
            X_train_shuffled, y_train_shuffled = X_train[:, permutation], y_train[:, permutation]
            for i in range(0, X_train.shape[1], batch_size):
                X_batch, y_batch = X_train_shuffled[:, i:i+batch_size], y_train_shuffled[:, i:i+batch_size]
                Y_pred_batch, cache = self.forward(X_batch)
                grads = self.backward(y_batch, cache)
                self.update_params(grads, learning_rate)

            train_pred, _ = self.forward(X_train)
            train_loss = compute_loss(y_train, train_pred)
            train_acc = compute_accuracy(y_train, train_pred)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            log_msg = (f"Epoch {Colors.BOLD}{epoch:03d}/{epochs}{Colors.NC} - "
                       f"loss: {Colors.GREEN}{train_loss:.4f}{Colors.NC} - "
                       f"acc: {Colors.GREEN}{train_acc:.4f}{Colors.NC}")

            if X_val is not None and y_val is not None:
                val_pred, _ = self.forward(X_val)
                val_loss = compute_loss(y_val, val_pred)
                val_acc = compute_accuracy(y_val, val_pred)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                log_msg += (f" - val_loss: {Colors.YELLOW}{val_loss:.4f}{Colors.NC} - "
                            f"val_acc: {Colors.YELLOW}{val_acc:.4f}{Colors.NC}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = copy.deepcopy(self.params)
                    epochs_no_improve = 0
                    best_epoch = epoch
                else:
                    epochs_no_improve += 1
                
                if patience > 0 and epochs_no_improve >= patience:
                    if verbose:
                        print(log_msg) # Print the final log before stopping
                        print(f"\n{Colors.YELLOW}--- Early Stopping Triggered ---{Colors.NC}")
                        print(f"Validation loss has not improved for {patience} epochs.")
                        print(f"Restoring model to best weights from epoch {Colors.BOLD}{best_epoch}{Colors.NC} (val_loss: {best_val_loss:.4f}).")
                    self.params = best_params
                    break
            
            if verbose:
                print(log_msg)
        
        if X_val is not None:
            self.params = best_params
        return history

    def predict(self, X):
        Y_pred, _ = self.forward(X)
        return Y_pred

    def evaluate(self, X, y):
        Y_pred, _ = self.forward(X)
        loss = compute_loss(y, Y_pred)
        accuracy = compute_accuracy(y, Y_pred)
        return loss, accuracy

    def save_model(self, path, preprocessor, history):
        """Saves model parameters, preprocessor state, and training history."""
        model_data = {
            'params': self.params, 'layer_dims': self.layer_dims, 
            'activations': self.activations[:-1],
            'preprocessor': preprocessor, # Save the entire preprocessor object
            'history': history
        }
        np.savez(path, **model_data)

    @staticmethod
    def load_model(path):
        """Loads a model, preprocessor, and history from a file."""
        data = np.load(path, allow_pickle=True)
        model = MLP(data['layer_dims'], data['activations'])
        model.params = data['params'].item()
        preprocessor = data['preprocessor'].item()
        history = data['history'].item()
        return model, preprocessor, history