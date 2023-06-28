import numpy as np
import pandas as pd
import pickle

# Load the dataset into a pandas DataFrame
data = pd.read_csv('bmi.csv')

# Extract the weight, height, and gender columns
weights = data['Weight'].values
heights = data['Height'].values
genders = data['Gender'].values


# Normalize the weight and height data
weights = (weights - weights.mean()) / weights.std()
heights = (heights - heights.mean()) / heights.std()

# Calculate the mean and standard deviation of the weight and height data
WEIGHT_MEAN = weights.mean()
WEIGHT_STD = weights.std()
HEIGHT_MEAN = heights.mean()
HEIGHT_STD = heights.std()

# Convert the gender data to numerical values (0 for Male, 1 for Female)
genders = (genders == 'Female').astype(int)

def sigmoid(x, deriv=False):
    if deriv:
        return x*(1-x)
    return 1/(1+np.exp(-x))

def mse_loss(y_true, y_pred): #mean squared error loss function
    return ((y_true - y_pred)**2).mean()

class ournn:
    def __init__(self):
        # Weights
        self.weights1 = np.random.rand(2, 5) # 2 input nodes, 5 hidden nodes
        self.weights2 = np.random.rand(5, 1) # 5 hidden nodes, 1 output node

        # Biases
        self.bias1 = np.random.rand(1, 5) # 1 bias for each hidden node
        self.bias2 = np.random.rand(1, 1) # 1 bias for the output node
    def feedforward(self, x):
        # x is a numpy array with 2 elements.
        hidden = sigmoid(np.dot(x, self.weights1) + self.bias1)
        output = sigmoid(np.dot(hidden, self.weights2) + self.bias2)
        return output
    def train(self, data, all_y_trues, learning_rate=0.1, epochs=10000):
        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # --- Feedforward ---
                hidden = sigmoid(np.dot(x, self.weights1) + self.bias1)
                output = sigmoid(np.dot(hidden, self.weights2) + self.bias2)

                # --- Backpropagation ---
                # Calculate the partial derivatives of the loss with respect to each weight and bias
                dL_doutput = -2 * (y_true - output)
                doutput_dhidden = self.weights2.T * sigmoid(output, deriv=True)
                dL_dhidden = dL_doutput * doutput_dhidden
                dhidden_dweights1 = x.reshape(-1, 1) * sigmoid(hidden, deriv=True)
                dL_dweights1 = dL_dhidden * dhidden_dweights1
                dL_dbias1 = dL_dhidden

                doutput_dweights2 = hidden.T
                dL_dweights2 = doutput_dweights2 * dL_doutput
                dL_dbias2 = dL_doutput

                # Update the weights and biases
                self.weights1 -= learning_rate * dL_dweights1
                self.bias1 -= learning_rate * dL_dbias1.sum(axis=0, keepdims=True)
                self.weights2 -= learning_rate * dL_dweights2
                self.bias2 -= learning_rate * dL_dbias2.sum()


            # Calculate the loss at the end of each epoch
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))
    def predict(self, weight, height):
        # Normalize the weight and height using the same normalization values as during training
        weight = (weight - WEIGHT_MEAN) / WEIGHT_STD
        height = (height - HEIGHT_MEAN) / HEIGHT_STD

        # Create a numpy array for the input data
        x = np.array([weight, height])

        # Feed the input data through the neural network
        output = self.feedforward(x)

        # The predicted gender is 1 if the output is greater than 0.5, and 0 otherwise
        predicted_gender = int(output > 0.5)

        # The predicted accuracy is the absolute value of the difference between the output and 0.5
        predicted_accuracy = abs(output - 0.5) # a higher accuracy will be closer to 0.5

        return predicted_gender, predicted_accuracy



# Combine the weight and height data into a single numpy array
X = np.column_stack((weights, heights))

# Use the gender data as the true outputs
y = genders

network = ournn()
network.train(X, y)

# Save the trained neural network to a file
with open('trained_network.pkl', 'wb') as f:
    pickle.dump(network, f)

# Load the trained neural network from a file
with open('trained_network.pkl', 'rb') as f:
    network = pickle.load(f)


while True:

    weight = float(input("Enter weight: "))
    height = float(input("Enter height: "))
    predicted_gender, predicted_accuracy = network.predict(weight, height)

    # Print the predicted gender and accuracy
    if predicted_gender == 0:
        print("Predicted gender: Male")
    else:
        print("Predicted gender: Female")
    print("Predicted accuracy: %.2f" % predicted_accuracy)