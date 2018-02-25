import numpy as np

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

# toy dataset
X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
                
y = np.array([[0],
			[1],
			[1],
			[0]])

print("Actual:")
print(y)

# randomize initial weights
np.random.seed(1234)
i_to_h1_weights = 2*np.random.random((3,4)) - 1
h1_to_h2_weights = 2*np.random.random((4,4)) - 1
h2_to_h3_weights = 2*np.random.random((4,4)) - 1
h3_to_o_weights = 2*np.random.random((4,1)) - 1

for j in range(2000):

    # forward pass
    input_layer = X
    hidden_layer_1 = tanh(np.dot(input_layer,i_to_h1_weights))
    hidden_layer_2 = tanh(np.dot(hidden_layer_1,h1_to_h2_weights))
    hidden_layer_3 = tanh(np.dot(hidden_layer_2,h2_to_h3_weights))
    output_layer = sigmoid(np.dot(hidden_layer_3,h3_to_o_weights))

    # track the change in error
    output_layer_error = y - output_layer    
    if j % 400 == 0:
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
    h3_to_o_weights += hidden_layer_3.T.dot(output_layer_delta)
    h2_to_h3_weights += hidden_layer_2.T.dot(hidden_layer_3_delta)
    h1_to_h2_weights += hidden_layer_1.T.dot(hidden_layer_2_delta)
    i_to_h1_weights += input_layer.T.dot(hidden_layer_1_delta)

print("Trained Output:")
print(output_layer)
