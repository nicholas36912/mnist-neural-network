from sklearn.datasets import load_digits
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

data = load_digits()
X = data.data 
y = data.target 
 
# Split data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,)

# Algorithm to one-hot encode the target labels
def one_hot_encode(labels, num_classes):
    one_hot_labels = np.zeros((len(labels), num_classes))
    for i in range(len(labels)):
        one_hot_labels[i, labels[i]] = 1
    return one_hot_labels

num_classes = len(np.unique(y_train))

y_train_one_hot = one_hot_encode(y_train, num_classes)
y_test_one_hot = one_hot_encode(y_test, num_classes)

class Layer_Dense:
    def __init__(self, n_input, n_neurons, n_classes, lr, num_iterations): 
        self.lr =lr
        self.num_iterations = num_iterations
        # Initialize weights and biases for the hidden layer using he initialization
        self.weights = np.random.randn(n_input, n_neurons) * np.sqrt(2/n_input)
        self.biases = np.zeros((1,n_neurons))   
        # Initialize weights and biases for the output layer using he initialization     
        self.weights2 = np.random.randn(n_neurons, n_classes) * np.sqrt(2/n_input)
        self.biases2 = np.zeros((1, n_classes))
        
    """
    The forward method computes the output of the layer and applies
    ReLU activation function for the hidden layer and softmax for 
    the output layer.
    """
    
    def forward(self, inputs):
        self.L1output = np.dot(inputs, self.weights) + self.biases
        self.A1output = Activation_RelU().forward(self.L1output)
        self.output = np.dot(self.A1output, self.weights2) + self.biases2
        self.predictions = Activation_Softmax().forward(self.output)
        self.input = inputs
        return self.predictions
        
    """
    The backward method computes gradients and updates weights and
    biases using gradient descent.
    """
    def backward(self, y):
        m = y.shape[0]
        dZ2 = self.predictions - y
        dW2 = np.dot(self.A1output.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dA2 = np.dot(dZ2, self.weights2.T)
        dA2[self.L1output <= 0] = 0 

        dZ1 = dA2
        dW1 = np.dot(self.input.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.weights2 -= self.lr * dW2
        self.biases2 -= self.lr * db2
        self.weights -= self.lr * dW1
        self.biases -= self.lr * db1

        return dA2, dW1, db1, dZ2, dW2, db2
    

class Activation_RelU:
    def forward(self, inputs):
        return np.maximum(0, inputs)


class Activation_Softmax:
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output


#cross entropy loss function
def loss_function(y_pred, y_true):
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / m  # adding a small value to prevent log(0)
    return loss
#calculate accuracy
def accuracy(y_pred, y_true):
    correct = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))
    total = y_true.shape[0]
    return correct / total
        
# Train the model       
def train(X, y, layer, loss, lr, num_iterations):
    for i in range(num_iterations):
        # Forward pass
        predictions = layer.forward(X)
        
        # Compute loss
        loss_value = loss(predictions, y)
        
        # Backward pass
        dA2, dW1, db1, dZ2, dW2, db2 = layer.backward(y)
                
# test the model     
def test(X, layer):
    predictions = layer.forward(X)
    return predictions

# Train and test the model multiple times
def train_test(X_train, X_test, y_train, y_test, num_runs=10):
    test_accuracies = []
    # run the model multiple times
    for i in range(num_runs):    
        layer = Layer_Dense(n_input=X_train.shape[1], n_neurons=30, n_classes=num_classes, lr=0.01, num_iterations=1000) #modify neuron count here
        train(X_train, y_train, layer, loss_function, 0.01, 1000) # modify learning rate and num_iterations here

        # Test the model
        test_predictions = test(X_test, layer)
        test_accuracy = accuracy(test_predictions, y_test)
        test_accuracies.append(test_accuracy)
        
        # Print accuracy of each run
        print(f"Run {i+1}: Test Accuracy = {test_accuracy}")
        
        
        """
        commented section to print the first 10 predictions and thier actual labels for each run.
        plots any incorrect predictions within, if any.
        """
        
        #   #Print subset of guesses for each run(10)
        # guesses = np.argmax(test_predictions[:10], axis=1)
        # true_labels = np.argmax(y_test[:10], axis=1)
        # print("Guesses for Run", i+1)
        # for j in range(10):
        #     print(f"Sample {j+1}: Predicted = {guesses[j]}, True = {true_labels[j]}")
        #     #plot any incorrect predictions
        #     if guesses[j] != true_labels[j]:
        #         plt.imshow(X_test[j].reshape(8, 8), cmap='gray')
        #         plt.title(f"Sample {j+1}: Predicted={guesses[j]}, True={true_labels[j]}")
        #         plt.show()
            
    return test_accuracies

# define number of runs
num_runs = 10
test_accuracies = train_test(X_train, X_test, y_train_one_hot, y_test_one_hot, num_runs)

# Calculate mean and standard deviation
mean_accuracy = np.mean(test_accuracies)
std_accuracy = np.std(test_accuracies)

print(f"Mean Test Accuracy: {mean_accuracy}")
print(f"Standard Deviation of Test Accuracy: {std_accuracy}")