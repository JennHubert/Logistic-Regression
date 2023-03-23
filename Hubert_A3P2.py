import pandas as pd
import numpy as np
# Imports above


def sigmoid(x): # Sigmoid function
    return 1/(1+np.exp(-x)) # Calculates sigmoid value and returns it
    

class LogisticRegression(): # regression class object

        def __init__(self, lr = 0.001, n_iters = 1000): # init constructor 
            self.lr = lr # Initializing learning rate
            self.n_iters = n_iters # Initializing number of iterations
            self.weights = None # Initializing weights
            self.bias = None # Initializing bias
             
        def fit(self, X, y): # Fit / Training function
            n_samples, n_features = X.shape # Finding the shape
            self.weights = np.zeros(n_features)  # Making the weights start at zero
            self.bias = 0 # Starting the bias at 0
            
            for _ in range(self.n_iters): #for number of iterations
                linear_p = np.dot(X, self.weights) + self.bias # Calculates linear predictions
                predictions = sigmoid(linear_p) # Sends linear prediction to sigmoid function
            
                dw = (1/n_samples) * np.dot(X.T, (predictions-y)) # Calculating the gradient for the weights
                db = (1/n_samples) * np.sum(predictions-y) # Calculationg the gradient for the bias
                
                self.weights = self.weights - self.lr*dw # Updating the weights
                self.bias = self.bias - self.lr*db # Updating the bias
                
                
            print("Coefficients (i.e. weights)") # Labeling output
            print(self.weights) # Printing the coefficients
            
            
        def predict(self, X): # Function making predictions !
            linear_p = np.dot(X, self.weights) + self.bias # Calculates linear predictions
            y_predictions = sigmoid(linear_p) # Sends linear prediction to sigmoid function
            
            # In the below code I explored the prediction values to better determine the decision boundary
            '''
            print("PREDICTIONS")
            print(y_predictions)
            print("MAX")
            print(y_predictions.max())
            print("MIN")
            print(y_predictions.min())
            '''
            
            class_predictions = [0 if y<=0.465 else 1 for y in y_predictions] # Converting into predictions, 0.465 as the decision boundary
            return class_predictions # Returning the predictions
    