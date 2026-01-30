# MNIST Digit Recognizer (Pytorch)
A simple sequential neural network built to recognize handwritten digits, specifically from the MNIST dataset.

## Project Overview
* Fully connected neural network trained using the MNIST dataset
* Utilizes the Cross Entropy Loss function and the ReLU activation function
* Demonstrates strong accuracy (96.1% success rate)

## Model Architechture
* Input Layer: 28x28 grayscale images flattened to a vector of 784 values
* Hidden Layer: 256 Neurons + ReLU activation
* Output layer: 10 neurons representative of certainty for each digit (0-9)

## Training Details
* Loss function: CrossEntropyLoss
* Loss decreases from ~2.3 -> <0.3 within first epoch
* 4 epochs to ensure a high level of accuracy

## Results
* Fast Convergence
* Stable training
* High accuracy on Kaggle [MNIST digit recognizer competition](https://www.kaggle.com/competitions/digit-recognizer) test data (public score: 0.96146)

## Future Improvements
* Incorporate a convolutional neural network model and observe differences in accuracy
* Develop interactive section displaying the accuracy in a digestible and enjoyable way

### Acknowledgments
* [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) I am aware the link leads to an empty page, however credit is due to Yann Lecun for the development
* A download to the dataset can be found on [Kaggle](https://www.kaggle.com/c/digit-recognizer/data)
* [Pytorch](https://pytorch.org/)
