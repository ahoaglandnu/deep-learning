import numpy as np
import matplotlib.pyplot as plt 
plt.style.use('ggplot')
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

raw_X,y = make_moons(n_samples=1000, noise=0.05, random_state=4321)

plt.figure(figsize=(8, 8))
plt.scatter(raw_X[:, 0], raw_X[:, 1], c=y, cmap='bwr')
plt.title('Two classes that cannot be separated by a line')
plt.savefig('moons', format='png')

# Set mean to 0 and stdev to 1
X = StandardScaler().fit_transform(raw_X)
y = np.reshape(y, (1000,1)) 

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
i_to_h1_weights = 2*np.random.random((2,8)) - 1 # 2 inputs to 8 neurons
h1_to_h2_weights = 2*np.random.random((8,8)) - 1 # 8 neurons to 8 neurons
h2_to_h3_weights = 2*np.random.random((8,8)) - 1 # 8 neurons to 8 neurons
h3_to_o_weights = 2*np.random.random((8,1)) - 1 # 8 neurons to 1 output

# try 3 different learning rates
learning_rates = [0.001, 0.01, 0.1]
for learning_rate in learning_rates:
    print("Learning rate:", learning_rate)

    for j in range(1000):

        # forward pass
        input_layer = X
        hidden_layer_1 = tanh(np.dot(input_layer,i_to_h1_weights))
        hidden_layer_2 = tanh(np.dot(hidden_layer_1,h1_to_h2_weights))
        hidden_layer_3 = tanh(np.dot(hidden_layer_2,h2_to_h3_weights))
        output_layer = sigmoid(np.dot(hidden_layer_3,h3_to_o_weights))

        # track the change in error
        output_layer_error = y - output_layer    
        if j % 100 == 0:
            print("Error:" + str(np.mean(np.abs(output_layer_error))))
        
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

    y_true = np.reshape(y, (1000,)) 
    y_pred = np.reshape(output_layer, (1000,))
    y_pred = np.round(y_pred, decimals=0)
    print(classification_report(y_true, y_pred))
    plt.figure(figsize=(8, 8))
    plt.scatter(raw_X[:, 0], raw_X[:, 1], c=y_pred, cmap='winter')
    plt.title('Results for learning rate', loc='left')
    plt.title(learning_rate)
    plt.savefig(str(learning_rate), format='png')

