# cs6910_assignment1

This repository contains the code base for Assignment 1 for the course CS6910: Deep Learning 2023. 


The problem statement calls for creating and training a "plain vanilla" feed-forward neural network from start, mainly using Python's Numpy package. 
The problem statement can be looked [here](https://wandb.ai/berserank/CS6910%20Assignment%201/reports/CS6910-Assignment-1--VmlldzozNzM0MTc4?accessToken=dmmeww9v6vdgsduqvh993yw1q821vpf18kjg2xj1z4ltgepapth6fkoyudkyfry8)

## Codebase structure: 
* __network.py__ - This file contains the code for the network.
    - I have created a general parent class `Layer` for all the layers in the network, all the layers inherit from this class. 
    - There are five layer classes that does different functionalities: 
        1. `LinearLayer` - This is the linear layer that performs the matrix multiplication and addition of bias.
        2. `ActivationLayer` - This is the activation layer that performs the activation function on the input.
        3. `SoftmaxOutputLayer` - This is the output layer that performs the softmax function on the input.
        4. `DropoutLayer` - This is the dropout layer that performs the dropout with a given probability.
        5. `BatchNormLayer` - This is the batch normalization layer.

* __feedforward.py__ - This file contains the `FeedForwardNet` that does the forward pass and backpropagation through the network.

* __train.py__ - This file contains the `train` function that trains the network. 

* __optimisers.py__ - This file contains the optimisers that are used to update the weights of the network.

* __activations.py__ - This file contains the activation functions that are used in the network.

* __loss.py__ - Contains the `CrossEntropy` and `MSE` loss functions.

## Running the file
The training loop is defined in the `train.py` file. It supports the following argument:

|Name                  |Default Value 	|Description| 
|----------------------|----------------|-----------|
|-wp, --wandb_project| 	myprojectname 	|Project name used to track experiments in Weights & Biases dashboard|
|-we, --wandb_entity| 	myname  	      |Wandb Entity used to track experiments in the Weights & Biases dashboard.|
|-d, --dataset| 	      fashion_mnist 	|choices: ["mnist", "fashion_mnist"]|
|-e, --epochs| 	        1 	            |Number of epochs to train neural network.|
|-b, --batch_size| 	    4 	            |Batch size used to train neural network.|
|-l, --loss| 	          cross_entropy 	|choices: ["mean_squared_error", "cross_entropy"]|
|-o, --optimizer| 	    sgd 	          |choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]|
|-lr, --learning_rate| 	0.1 	          |Learning rate used to optimize model parameters|
|-m, --momentum| 	      0.5 	          |Momentum used by momentum and nag optimizers.|
|-beta, --beta| 	      0.5 	          |Beta used by rmsprop optimizer|
|-beta1, --beta1| 	    0.5 	          |Beta1 used by adam and nadam optimizers.|
|-beta2, --beta2| 	    0.5 	          |Beta2 used by adam and nadam optimizers.|
|-eps, --epsilon| 	    0.000001 	      |Epsilon used by optimizers.|
|-w_d, --weight_decay| 	.0 	            |Weight decay used by optimizers.|
|-w_i, --weight_init| 	random 	        |choices: ["random", "Xavier"]|
|-nhl, --num_layers| 	  1 	            |Number of hidden layers used in feedforward neural network.|
|-sz, --hidden_size| 	  4 	            |Number of hidden neurons in a feedforward layer.|
|-a, --activation| 	    sigmoid 	      |choices: ["identity", "sigmoid", "tanh", "ReLU"]|
|-sc, --scaling| 	      standard 	      |choices: ["standard", "minmax"]|
|-w, --wandb| 	        False 	        |Use wandb to log metrics|
|-que, --question|      None            |The Question Number you want to run.|
|-do, --dropout_rate| 	0.0 	          |Dropout rate.|
|-bn, --batch_norm| 	  False 	        |Use batch normalization.|

A particular file can be run through: 
```
python filename.py
```

To run a particular question i of the assignment, simply run: 
```
python train.py -que i
```  
This will run the file with the default arguments which can be changed through command line.

