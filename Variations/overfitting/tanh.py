import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# load dataset
data = load_breast_cancer()

# Prep dataset
raw_X = data.data
y = data.target
#print(raw_X.shape) # needed shape to set below
#print(y.shape)

# recommended upper bound of hidden neurons
samples = len(y)

def neuro(samples):
    neurons = samples / (2 * (30/1))
    return neurons

print("Recommended number of hidden neurons:", neuro(samples))

# Set mean to 0 and stdev to 1
X = StandardScaler().fit_transform(raw_X)
y = np.reshape(y, (569,1)) 

# activation functions
def sigmoid(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

def tanh(x,deriv=False):
    if(deriv==True):
        z = np.tanh(x)
        return 1 - (z**2)
    return np.tanh(x)

# randomize initial weights
np.random.seed(1234)
i_to_h1_weights = 2*np.random.random((30,3)) - 1 # 30 inputs to 3 neurons
h1_to_h2_weights = 2*np.random.random((3,3)) - 1 # 3 neurons to 3 neurons
h2_to_h3_weights = 2*np.random.random((3,3)) - 1 # 3 neurons to 3 neurons
h3_to_o_weights = 2*np.random.random((3,1)) - 1 # 3 neurons to 1 output

# randomize initial biases
np.random.seed(123)
hidden_layer_1_bias = 2*np.random.random((3)) - 1 # bias for 3 neurons
hidden_layer_2_bias = 2*np.random.random((3)) - 1 # bias for 3 neurons
hidden_layer_3_bias = 2*np.random.random((3)) - 1 # bias for 3 neurons
output_layer_bias = 2*np.random.random((1)) - 1 # bias for output

# set learning rate
learning_rate = 0.003
print("Learning rate:", learning_rate)

# epochs
epochs = 10000
print(epochs, "epochs")
for j in range(epochs):

    # forward pass
    input_layer = X 
    hidden_layer_1 = tanh(np.dot(input_layer,i_to_h1_weights) + hidden_layer_1_bias)
    hidden_layer_2 = tanh(np.dot(hidden_layer_1,h1_to_h2_weights) + hidden_layer_2_bias)
    hidden_layer_3 = tanh(np.dot(hidden_layer_2,h2_to_h3_weights) + hidden_layer_3_bias)
    output_layer = sigmoid(np.dot(hidden_layer_3,h3_to_o_weights) + output_layer_bias)

    # track the change in error
    output_layer_error = y - output_layer    
    if j % 1000 == 0:
        print("Loss:" + str(np.mean(np.abs(output_layer_error))))
    
    # back propagate    
    output_layer_delta = output_layer_error * sigmoid(output_layer,deriv=True)
    
    hidden_layer_3_error = output_layer_delta.dot(h3_to_o_weights.T)
    hidden_layer_3_delta = hidden_layer_3_error * tanh(hidden_layer_3,deriv=True)

    hidden_layer_2_error = hidden_layer_3_delta.dot(h2_to_h3_weights.T)
    hidden_layer_2_delta = hidden_layer_2_error * tanh(hidden_layer_2,deriv=True)

    hidden_layer_1_error = hidden_layer_2_delta.dot(h1_to_h2_weights.T)
    hidden_layer_1_delta = hidden_layer_1_error * tanh(hidden_layer_1,deriv=True)

    # update weights
    h3_to_o_weights += learning_rate * hidden_layer_3.T.dot(output_layer_delta)
    h2_to_h3_weights += learning_rate * hidden_layer_2.T.dot(hidden_layer_3_delta)
    h1_to_h2_weights += learning_rate * hidden_layer_1.T.dot(hidden_layer_2_delta)
    i_to_h1_weights += learning_rate * input_layer.T.dot(hidden_layer_1_delta)

    # update biases
    output_layer_bias = output_layer_delta[-1]
    hidden_layer_3_bias = hidden_layer_3_delta[-1] 
    hidden_layer_2_bias = hidden_layer_2_delta[-1] 
    hidden_layer_1_bias = hidden_layer_1_delta[-1]

y_true = np.reshape(y, (569,)) 
y_pred = np.reshape(output_layer, (569,))
y_pred = np.round(y_pred, decimals=0)
print(classification_report(y_true, y_pred))
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
print("tn fp fn tp")
print(tn, fp, fn, tp)