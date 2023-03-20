import numpy as np 
from network import Layer, LinearLayer, SoftmaxOutputLayer, ActivationLayer, DropoutLayer, BatchNormLayer
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import wandb

class FeedForwardNet:
    def __init__(self, layers, use_wandb=False) -> None:
        self.layers = layers
        self.use_wandb = use_wandb
        pass 
    
    def forward(self, input_samples, mode='train'):
        """Performs forward propagation over all the layers and returns the activations"""
        self.activations = [input_samples]
        # Compute forward activations for each layer
        X = input_samples
        for layer in self.layers:
            if isinstance(layer, LinearLayer):
                Y = layer.get_output(X, mode)
            else:
                Y = layer.get_output(Y, mode)
                self.activations.append(Y)
                X = self.activations[-1]
        return self.activations

    def backward(self, targets):
        output_grad = None
        # Propagate the error backwards through the network
        for layer in reversed(self.layers):
            if isinstance(layer, LinearLayer):
                # If the layer is a linear layer
                layer.get_params_grad(X, input_grad)
                output_grad = layer.get_input_grad(Y, input_grad)
                # Y = self.activations.pop()
            else:
                Y = self.activations.pop()
                # The output layer error is a special case
                if output_grad is None:
                    input_grad = layer.get_input_grad(Y, targets)
                else:
                    input_grad = layer.get_input_grad(Y, output_grad)
            X = self.activations[-1]



    def train(self, X_train, y_train, X_val, y_val, epochs, optimiser, loss, batch_size=32):
        # Create batches (X, Y) for training
        num_batches = X_train.shape[0] // batch_size
        XT_batches = list(zip(
            np.array_split(X_train, num_batches, axis=0),
            np.array_split(y_train, num_batches, axis=0)
        ))
        
        # Training loop
        batch_costs = []  # List of costs for each batch
        epoch_train_costs = []  # List of train costs for each epoch
        val_costs = []  # List of validation costs for each epoch
        val_accuracies = []  # List of validation accuracies for each epoch

        for epoch in range(epochs):
            for X, T in tqdm(XT_batches):
                # Forward propagation
                activations = self.forward(X, mode='train')
                # Compute the cost
                batch_cost = self.layers[-1].get_cost(activations[-1], T, loss=loss)
                batch_costs.append(batch_cost)
                # Compute the gradients
                self.backward(T)
                # Update the parameters
                optimiser.update_params(self.layers)
            # Compute the cost on the training set
            activations = self.forward(X_train, mode='train')
            epoch_train_cost = np.sum(batch_costs) / len(batch_costs)
            epoch_train_costs.append(np.sum(batch_costs)/len(batch_costs))
            epoch_train_accuracy = accuracy_score(np.argmax(activations[-1], axis=1), np.argmax(y_train, axis=1))

            # Compute the cost and accuracy on the vaidation set
            activations = self.forward(X_val, mode='test')
            val_cost = self.layers[-1].get_cost(activations[-1], y_val, loss=loss)
            val_costs.append(val_cost)
            val_accuracy = accuracy_score(np.argmax(activations[-1], axis=1), np.argmax(y_val, axis=1))
            val_accuracies.append(val_accuracy)

            if self.use_wandb:
                wandb.log({
                    "epoch": epoch, \
                    "train_cost": epoch_train_cost, \
                    "train_accuracy": epoch_train_accuracy, \
                    "val_cost": val_cost, \
                    "val_accuracy": val_accuracy
                })

            # Print the cost and accuracy for each epoch
            print(f"Epoch {epoch+1}/{epochs} - Train cost: {epoch_train_costs[-1]:.4f} - Train accuracy: {epoch_train_accuracy} - Val cost: {val_costs[-1]:.4f} - Val accuracy: {val_accuracies[-1]:.4f}")
        return epoch_train_costs, epoch_train_accuracy, val_costs, val_accuracies

    def predict(self, X):
        """Returns the predictions for the given input samples"""
        out = self.forward(X, mode='test')[-1]
        return out

