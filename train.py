import numpy as np 
import pandas as pd
import seaborn as sns
import wandb
import yaml
from keras.datasets import fashion_mnist, mnist
import matplotlib.pyplot as plt
import argparse
from network import Layer, LinearLayer, SoftmaxOutputLayer, ActivationLayer, DropoutLayer, BatchNormLayer
from activations import Sigmoid, Softmax
from loss import CrossEntropy, MSE
from tqdm import tqdm
from optimiser import SGD, MGD, NAG, Adam, RMSProp, Nadam, AdaGrad
from feedforward import FeedForwardNet
from sklearn.metrics import confusion_matrix, accuracy_score

def data(dataset, scaling):
    if dataset == 'mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    elif dataset == 'fashion_mnist':
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError('Invalid dataset')
    
    # Remove nan values 
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)
    y_train = np.nan_to_num(y_train)
    y_test = np.nan_to_num(y_test)
    
    # split the train set into train and validation set
    X_val, y_val = X_train[40000:], y_train[40000:]
    X_train, y_train = X_train[:40000], y_train[:40000]

    if scaling == "minmax":
        # do min-max scaling
        x_max_tr = np.max(X_train)
        x_min_tr = np.min(X_train)
        x_max_ts = np.max(X_test)
        x_min_ts = np.min(X_test)
        X_train = (X_train - x_min_tr) / (x_max_tr - x_min_tr)
        X_val = (X_val - x_min_tr) / (x_max_tr - x_min_tr)
        X_test = (X_test - x_min_ts) / (x_max_ts - x_min_ts)

    elif scaling == "standard":
        # Rescale the images from [0, 255] to [0, 1]
        X_train = X_train / 255.0
        X_val = X_val / 255.0
        X_test = X_test / 255.0

    # Reshape data to be of shape (num_samples, num_features)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Convert labels to one-hot vectors
    y_train = np.eye(10)[y_train]
    y_val = np.eye(10)[y_val]
    y_test = np.eye(10)[y_test]
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def get_accuracy(predictions, targets):
    """Returns the accuracy of the predictions"""
    return np.sum(np.argmax(predictions, axis=1) == np.argmax(targets, axis=1)) / targets.shape[0]

def train():
    X_train, y_train, X_val, y_val, X_test, y_test = data(args.dataset, args.scaling)
    print(f"X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}")

    layers = []
    layers.append(LinearLayer(X_train.shape[1], args.hidden_size, args.weight_init))
    layers.append(ActivationLayer(args.activation))
    if args.dropout_rate > 0:
        layers.append(DropoutLayer(args.dropout_rate))
    if args.batch_norm:
        layers.append(BatchNormLayer(gamma=1, beta=0, momentum=0.9))
    for i in range(args.num_layers):
        layers.append(LinearLayer(args.hidden_size, args.hidden_size, args.weight_init))
        layers.append(ActivationLayer(args.activation))
        if args.dropout_rate > 0:
            layers.append(DropoutLayer(args.dropout_rate))
        if args.batch_norm:
            layers.append(BatchNormLayer(gamma=1, beta=0, momentum=0.9))
    layers.append(LinearLayer(args.hidden_size, y_train.shape[1], args.weight_init))
    layers.append(SoftmaxOutputLayer(args.weight_decay))

    optim_params = {
        "sgd": [args.learning_rate],
        "momentum": [args.learning_rate, args.momentum],
        "nag": [args.learning_rate, args.momentum],
        "rmsprop": [args.learning_rate, args.beta, args.epsilon],
        "adam": [args.learning_rate, args.beta1, args.beta2, args.epsilon],
        "nadam": [args.learning_rate, args.beta1, args.beta2, args.epsilon]
    }

    if args.optimiser == "momentum":
        optimiser = MGD(*optim_params[args.optimiser])
    elif args.optimiser == "nag":
        optimiser = NAG(*optim_params[args.optimiser])
    elif args.optimiser == "adagrad":
        optimiser = AdaGrad(*optim_params[args.optimiser])
    elif args.optimiser == "rmsprop":
        optimiser = RMSProp(*optim_params[args.optimiser])
    elif args.optimiser == "adam":
        optimiser = Adam(*optim_params[args.optimiser])
    elif args.optimiser == "nadam":
        optimiser = Nadam(*optim_params[args.optimiser])
    else:
        optimiser = SGD(*optim_params[args.optimiser])

    model = FeedForwardNet(layers)
    epoch_train_costs, epoch_train_accuracy, val_costs, val_accuracies = model.train(X_train, y_train, X_val, y_val, epochs=args.epochs, batch_size=args.batch_size, optimiser=optimiser, loss=args.loss)
    # test the model
    test_cost, test_accuracy = test(model, X_test, y_test, num_batches=10)
    print(f"Test Cost: {test_cost}, Test Accuracy: {test_accuracy}")    


# test the model 
def test(model, X_test, y_test, num_batches):
    test_costs = []
    test_accuracies = []
    XT_batches = list(zip(
        np.array_split(X_test, num_batches, axis=0),  # X samples
        np.array_split(y_test, num_batches, axis=0)  # Y labels
    ))

    for X, T in tqdm(XT_batches): # Iterate over batches
        # Forward propagation
        activations = model.forward(X, mode="test")
        # Compute the cost
        batch_cost = CrossEntropy().calc_loss(activations[-1], T)
        test_costs.append(batch_cost)
        # Compute the accuracy
        # test_accuracy = get_accuracy(activations[-1], T)
        test_accuracy = accuracy_score(np.argmax(T, axis=1), np.argmax(activations[-1], axis=1))
        test_accuracies.append(test_accuracy)
    test_accuracy = np.mean(test_accuracies)
    return np.mean(test_costs), test_accuracy


sweep_config_1 = {
    "name": "Fashion MNIST Random Bayes Full",
    "method": "bayes", 
    "metric": {
        "name": "val_accuracy",
        "goal": "maximize"
    },
    "parameters": {
        "epochs": { 
            "values": [10, 20] 
        },
        "batch_size": {
            "values": [128, 256, 512]
        },
        "learning_rate": {
            "values": [0.001, 0.0001]
        },
        "optimiser": {
            "values": ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]
        },
        "loss": {
            "values": ["CrossEntropy", "mean_squared_error"]
        }, 
        "activation": {
            "values": ["Sigmoid", "tanh", "ReLU", "LeakyReLU", "elu"]
        },
        "num_hidden_layers": {
            "values": [3, 4, 5]
        },
        "hidden_layer_size": {
            "values": [32, 64, 128]
        },
        "scaling": {
            "values": ['minmax', 'standard']
        },
        "weight_initialisation": {
            "values": ["random", "xavier"]
        },
        "weight_decay": {
            "values": [0.001, 0]
        },
        "dropout_rate": {
            "values": [0.0, 0.1, 0.2]
        },
        "batch_norm": {
            "values": [True, False]
        }
    }
}

def train_nn(config = sweep_config_1):
    wandb.init()
    with wandb.init(config=config):
        config = wandb.config
        wandb.run.name = "e_{}_bs_{}_lr_{}_opt_{}_loss_{}_act_{}_nhl_{}_hls_{}_sc_{}_wi_{}_wd_{}_dr_{}_bn_{}".format(config.epochs, \
                                                                                            config.batch_size, \
                                                                                            config.learning_rate, \
                                                                                            config.optimiser, \
                                                                                            config.loss, \
                                                                                            config.activation, \
                                                                                            config.num_hidden_layers, \
                                                                                            config.hidden_layer_size, \
                                                                                            config.scaling, \
                                                                                            config.weight_initialisation, \
                                                                                            config.weight_decay, \
                                                                                            config.dropout_rate, \
                                                                                            config.batch_norm)

        X_train, y_train, X_val, y_val, X_test, y_test = data(dataset='fashion_mnist', scaling=config.scaling)

        layers = []
        layers.append(LinearLayer(X_train.shape[1], config.hidden_layer_size, config.weight_initialisation))
        layers.append(ActivationLayer(config.activation))
        if config.dropout_rate > 0:
            layers.append(DropoutLayer(config.dropout_rate))
        if config.batch_norm:
            layers.append(BatchNormLayer(gamma=1, beta=0, momentum=0.9))
        for _ in range(config.num_hidden_layers):
            layers.append(LinearLayer(config.hidden_layer_size, config.hidden_layer_size, config.weight_initialisation))
            layers.append(ActivationLayer(config.activation))
            if config.dropout_rate > 0:
                layers.append(DropoutLayer(config.dropout_rate))
            if config.batch_norm:
                layers.append(BatchNormLayer(gamma=1, beta=0, momentum=0.9))
        layers.append(LinearLayer(config.hidden_layer_size, y_train.shape[1], config.weight_initialisation))
        layers.append(SoftmaxOutputLayer(config.weight_decay))

        optim_params = {
                    "sgd": [config.learning_rate],
                    "momentum": [config.learning_rate, args.momentum],
                    "nag": [config.learning_rate, args.momentum],
                    "rmsprop": [config.learning_rate, args.beta, args.epsilon],
                    "adam": [config.learning_rate, args.beta1, args.beta2, args.epsilon],
                    "nadam": [config.learning_rate, args.beta1, args.beta2, args.epsilon]
                }

        if config.optimiser == "momentum":
            optimiser = MGD(*optim_params[config.optimiser])
        elif config.optimiser == "nag":
            optimiser = NAG(*optim_params[config.optimiser])
        elif config.optimiser == "rmsprop":
            optimiser = RMSProp(*optim_params[config.optimiser])
        elif config.optimiser == "adam":
            optimiser = Adam(*optim_params[config.optimiser])
        elif config.optimiser == "nadam":
            optimiser = Nadam(*optim_params[config.optimiser])
        elif config.optimiser == "sgd":   
            optimiser = SGD(*optim_params[config.optimiser])


        model = FeedForwardNet(layers, use_wandb=True)
        epoch_train_costs, epoch_train_accuracy, val_costs, val_accuracies = model.train(X_train, y_train, X_val, y_val, config.epochs, optimiser, config.loss, config.batch_size)
        test_cost, test_accuracy = test(model, X_test, y_test, num_batches=100)


        wandb.log({
            "epoch_train_costs": epoch_train_costs[-1], \
            "epoch_train_accuracy": epoch_train_accuracy, \
            "val_costs": val_costs[-1], \
            "val_accuracy": val_accuracies[-1], \
            "test_cost": test_cost, \
            "test_accuracy": test_accuracy, \
            "epoch": config.epochs
        })

if __name__ == "__main__":
    # Parse the command line arguments
    parser = argparse.ArgumentParser(description='Train a neural network')
    parser.add_argument('-wp', '--wandb_project', type=str, default='deep-learning-assignment-1', help='Wandb project name')
    parser.add_argument('-we', '--wandb_entity', type=str, default='chouhan-romil01', help='Wandb entity name')
    parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist', help='choices: ["mnist", "fashion_mnist"]')
    parser.add_argument('-e', '--epochs', type=int, default=1, help='Number of epochs to train neural network')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy', help='choices: ["mean_squared_error", "cross_entropy"]')
    parser.add_argument('-o', '--optimiser', type=str, default='sgd', help='choices: ["sgd", "momentum", "nag", "adagrad", "rmsprop", "adam", "nadam"]')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1, help='Learning rate used to optimize model parameters')
    parser.add_argument('-m', '--momentum', type=float, default=0.5, help='Momentum used by momentum and nag optimisers')
    parser.add_argument('-beta', '--beta', type=float, default=0.5, help='Beta used by rmsprop optimisers')
    parser.add_argument('-beta1', '--beta1', type=float, default=0.5, help='Beta1 used by adam and nadam optimisers')
    parser.add_argument('-beta2', '--beta2', type=float, default=0.5, help='Beta2 used by adam and nadam optimisers')
    parser.add_argument('-eps', '--epsilon', type=float, default=1e-6, help='Epsilon used by optimisers')
    parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0, help='Weight decay used by optimisers')
    parser.add_argument('-w_i', '--weight_init', type=str, default='random', help='choices: ["random", "xavier"]')
    parser.add_argument('-nhl', '--num_layers', type=int, default=1, help='Number of hidden layers used in feedforward neural network')
    parser.add_argument('-sz', '--hidden_size', type=int, default=4, help='Number of hidden neurons used in a feedforward layer')
    parser.add_argument('-a', '--activation', type=str, default='sigmoid', help='choices: ["identity", "sigmoid", "tanh", "ReLU", "LeakyReLU", "elu"]')
    parser.add_argument('-sc', '--scaling', type=str, default='standard', help='choices: ["standard", "minmax"]')
    parser.add_argument('-w', '--wandb', type=bool, default=False, help='Use wandb to log metrics')
    parser.add_argument('-que', '--question', type=int, default=None, help='The Question Number you want to run')
    parser.add_argument('-do', '--dropout_rate', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('-bn', '--batch_norm', type=bool, default=False, help='Use batch normalization')
    args = parser.parse_args()

    
    # Load the dataset
    if args.dataset == 'mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    elif args.dataset == 'fashion_mnist':
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()    
    
    num_classes = 10

    if args.question:
        wandb.login(key="b3a089bfb32755711c3923f3e6ef67c0b0d2409b")
        with open("./sweep_config.yml", "r") as file:
            sweep_config = yaml.safe_load(file)

        #question 1
        if args.question == 1:
            wandb.init(project=args.wandb_project)
            wandb.run.name = "question-1"
            y_train_one_hot =  np.eye(num_classes)[y_train]
            x = np.concatenate(X_train, axis=0);y = np.concatenate(y_train_one_hot, axis=0)
            if args.dataset == "fashion_mnist":
                class_mapping = {0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"} 
            else:
                class_mapping = {i:str(i) for i in range(10)}
        
            plt.figure(figsize=(12, 5))
            img_list = []
            class_list = []

            for i in range(num_classes):
                position = np.argmax(y_train == i)
                image = X_train[position,:,:]
                plt.subplot(2, 5, i+1)
                plt.imshow(image)
                plt.title(class_mapping[i])
                img_list.append(image)
                class_list.append(class_mapping[i])
            wandb.log({"Question 1": [wandb.Image(img, caption=class_name) for img, class_name in zip(img_list, class_list)]})


        #question 2, 3
        elif args.question in [2,3]:
            train()
            

        #question 4
        elif args.question == 4:
            sweep_id = wandb.sweep(sweep_config_1, project="Fashion MNIST Sweep")
            wandb.agent(sweep_id, train_nn, count=100)
            
            

        elif args.question in [5, 6]:
            print("Please Check the Readme and my wandb assignment page")

        elif args.question == 7:
            X_train, y_train, X_val, y_val, X_test, y_test = data(args.dataset)   
            layers = []
            for i in range(args.num_layers):
                if i == 0: 
                    layers.append((LinearLayer(X_train.shape[1], args.hidden_size, args.weight_init), ActivationLayer(args.activation)))
                else: 
                    layers.append((LinearLayer(args.hidden_size, args.hidden_size, args.weight_init), ActivationLayer(args.activation)))
            layers.append((LinearLayer(args.hidden_size, y_train.shape[1], args.weight_init), SoftmaxOutputLayer(args.weight_decay)))

            X_train, y_train, X_val, y_val, X_test, y_test = data(args.dataset)

            optim_params = {
                "sgd": [args.learning_rate],
                "momentum": [args.learning_rate, args.momentum],
                "nag": [args.learning_rate, args.momentum],
                "rmsprop": [args.learning_rate, args.beta, args.epsilon],
                "adam": [args.learning_rate, args.beta1, args.beta2, args.epsilon],
                "nadam": [args.learning_rate, args.beta1, args.beta2, args.epsilon]
            }

            model = FeedForwardNet(layers)  
            # look for the best hyperparameters using the sweeps
            y_test_predicted = model.predict(X_test)
            cm = confusion_matrix(y_test, y_test_predicted)
            ### Confusion Matrix
            df_cm = pd.DataFrame(cm, index = range(1, len(cm)+1), columns = range(1, len(cm) + 1))
            plt.figure(figsize=(12, 5))
            tmp = sns.heatmap(df_cm, annot=True)
            fig = tmp.get_figure()
            fig.savefig("./confusion.png", dpi=400)
            plt.show()
        