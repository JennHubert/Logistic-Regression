import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from Class_Instance import LogisticRegression


penguins = pd.read_csv('/Users/jenniferhubert/DSSA Spring 2023/Machine Learning/penguins.csv') # import data

df = penguins[['CulmenLength_mm', 'CulmenDepth_mm', 'FlipperLength_mm', 'BodyMass_g', 'Species']] # Columns to be used

df = df.dropna() # Get rid of nulls
y = df.Species # Getting the y
y = np.where(y == 'Gentoo penguin (Pygoscelis papua)', 1, 0) # Encoding
df2 = df[['CulmenLength_mm', 'CulmenDepth_mm', 'FlipperLength_mm', 'BodyMass_g']] # the X values
X = df2 # the X values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Splitting data
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression() # Turning it into a variable
model.fit(X_train, y_train) # Trainging the model
y_pred = model.predict(X_test) # Making predictions



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # Import for the picture of the cm
import matplotlib.pyplot as plt # Import for cm

print("") # Empty line
from sklearn.metrics import classification_report # Importing the report
print(classification_report(y_test, y_pred)) # Printing the report this way because it is good for the report

print("") # Empty line - easier to read
print("CONFUSION MATRIX:") # Easier to read with label
print(confusion_matrix(y_test, y_pred)) # CM Scores

cm = confusion_matrix(y_test, y_pred) # CM Scores again
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=[1,0]) # Making the display
disp.plot() # Plotting it
plt.show() # Showing it




