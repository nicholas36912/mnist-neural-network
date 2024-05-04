# MNIST Classification Neural Network

This is a simple implementation of a neural network for classifying handwritten digits from the MNIST dataset. The neural network is implemented from scratch in Python.

## Requirements
- Python 3
- NumPy
- scikit-learn

## Implementation Details
- **scikit-learn**: used to initialize the dataset and split into training and testing sets

## One-Hot Encoding
- The target classification labels range from 0 to 9.
- One-hot encoding transforms these target labels into binary vectors of length 10.
- This ensures that the neural network can effectively understand and learn from these labels.

## Neural Network Architecture
- The neural network consists of one hidden layer with ReLU activation and one output layer with softmax activation.
- Training is performed using gradient descent with cross-entropy loss.

### Hidden Layer
- The hidden layer receives the pixel values of the input image.
- ReLU activation function is applied to help the network learn complex patterns in the data and converge faster during training.
- 20 neurons are used in this layer to maximize accuracy, although 10 produces similar results.

### Output Layer
- The output layer of the neural network consists of 10 neurons, each representing a digit from 0 to 9.
- Softmax activation function is applied to the output layer, producing a probability distribution.
- One-hot encoding of labels allows the network to learn these digits for each input image.

## Testing Process
- The MNIST dataset is split into training and testing sets.
- The model is trained on the training set.

## Training Parameters
- Learning Rate: 0.01
- Number of iterations: 1000
- Number of neurons in the hidden layer: 20

After training, the model is tested on the testing set to evaluate its accuracy. This process is repeated multiple times (in this case 10) to calculate the mean accuracy and standard deviation.

## Optimization
- Xavier initialization is used to optimize ReLU activation function in the hidden layer, avoiding gradient issues, and promoting faster convergence.
- This results in significant accuracy and consistency across runs.

Overall, the model achieves a mean test accuracy of 95%.

