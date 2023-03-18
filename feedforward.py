import numpy as np 
from keras.datasets import fashion_mnist, mnist
import matplotlib.pyplot as plt
from network import Layer, LinearLayer, SoftmaxOutputLayer, ActivationLayer
from activations import Sigmoid, Softmax
from loss import CrossEntropy, MSE
from tqdm import tqdm
from optimiser import SGD, MGD, NAG, Adam, RMSProp, Nadam, AdaGrad
from sklearn.metrics import accuracy_score

class FeedForwardNet:
    def __init__(self, layers) -> None:
        self.layers = layers
        pass 

    def forward(self, input_samples):
        """Performs forward propagation over all the layers and returns the activations"""
        self.activations = [input_samples]
        # Compute forward activations for each layer
        X = input_samples
        for linear, activation in self.layers:
            # Get the activation for the current layer
            Y = linear.get_output(X)
            # activations.append(Y)
            Y = activation.get_output(Y)
            # Store the output for future processing 
            self.activations.append(Y)
            X = self.activations[-1]
        return self.activations
    
    def backward(self, targets):
        output_grad = None
        # Propagate the error backwards through the network 
        for linear, activation in reversed(self.layers):
            Y = self.activations.pop()
            # The output layer error is a special case
            if output_grad is None:
                input_grad = activation.get_input_grad(Y, targets)
            else:
                input_grad = activation.get_input_grad(Y, output_grad)
            # Get the input of this layer (activations of the previous layer)
            X = self.activations[-1]
            # Compute the layer parameter gradients used to update the layer parameters 
            linear.get_params_grad(X, input_grad)
            # Propagate the parameter gradients through the linear layer 
            output_grad = linear.get_input_grad(Y, input_grad)

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
                activations = self.forward(X)
                # Compute the cost
                batch_cost = self.layers[-1][1].get_cost(activations[-1], T, loss=loss)
                batch_costs.append(batch_cost)
                # Compute the gradients
                self.backward(T)
                # Update the parameters
                optimiser.update_params(self.layers)
            # Compute the cost on the training set
            activations = self.forward(X_train)
            epoch_train_costs.append(np.sum(batch_costs)/len(batch_costs))
            # train_costs.append(train_cost)

            # Compute the cost and accuracy on the vaidation set
            activations = self.forward(X_val)
            val_cost = self.layers[-1][1].get_cost(activations[-1], y_val, loss=loss)
            val_costs.append(val_cost)
            val_accuracy = accuracy_score(np.argmax(activations[-1], axis=1), np.argmax(y_val, axis=1))
            # val_accuracy = self.get_accuracy(np.argmax(activations[-1], axis=1), np.argmax(y_val, axis=1))
            val_accuracies.append(val_accuracy)

            # Print the cost and accuracy for each epoch
            print(f"Epoch {epoch+1}/{epochs} - Train cost: {epoch_train_costs[-1]:.4f} - Val cost: {val_costs[-1]:.4f} - Val accuracy: {val_accuracies[-1]:.4f}")
        return epoch_train_costs, val_costs, val_accuracies

    def predict(self, X):
        """Returns the predictions for the given input samples"""
        out = self.forward(X)[-1]
        return out

    def get_accuracy(self, predictions, targets):
        """Returns the accuracy of the predictions"""
        return np.sum(predictions == targets) / len(targets)

