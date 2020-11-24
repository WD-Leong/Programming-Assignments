# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 02:59:17 2020

@author: admin
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def binary_ce(y, y_hat, eps=1.0e-3):
    # Clip the value for numerical stability. #
    y_mask = np.where(np.logical_or(
        y_hat <= eps, y_hat >= 1.0-eps), 0, 1)
    y_hat  = np.clip(y_hat, eps, 1.0-eps)
    
    ce_loss = y * np.log(y_hat)
    ce_loss = (1.0 - y) * np.log(1.0 - y_hat)
    ce_loss = -1.0 * np.mean(ce_loss)
    return ce_loss, y_mask

def relu(z):
    return np.maximum(0.0, z)
    
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def error_derivative(y, y_hat, eps=1.0e-6):
    y_hat = np.clip(y_hat, eps, 1.0-eps)
    
    dE = -1.0 * (
        np.divide(y, y_hat) -\
        np.divide(1.0 - y, 1.0 - y_hat))
    return dE

def sigmoid_derivative(z):
    return sigmoid(z) * (1.0 - sigmoid(z))

def relu_derivative(z):
    return np.where(z <= 0.0, 0.0, 1.0)

def forward_pass(nnet, X_input):
    layer_out = []
    in_layer  = X_input
    for n_layer in range(n_layers):
        activation  = "activation_" + str(n_layer)
        bias_name   = "bias_" + str(n_layer)
        weight_name = "layer_" + str(n_layer)
        
        tmp_output = nnet[bias_name] +\
            np.matmul(in_layer, nnet[weight_name])
        
        if nnet[activation] == "relu":
            tmp_output = relu(tmp_output)
        else:
            tmp_output = sigmoid(tmp_output)
        
        in_layer = tmp_output
        layer_out.append(tmp_output)
    return layer_out

def backward_pass(
    n_layers, nnet, X_train, y_train, nnet_pred, layer_out):
    # Backward propagation. #
    # Note that outermost layer is layer zero. #
    delta_l = 0.0
    for n_layer in range(n_layers-1, -2, -1):
        activation  = "activation_" + str(n_layer+1)
        bias_name   = "bias_" + str(n_layer+1)
        weight_name = "layer_" + str(n_layer+1)
        
        if n_layer == (n_layers-1):
            delta_j = error_derivative(y_train, nnet_pred)
            delta_j = delta_j * sigmoid_derivative(nnet_pred)
            
            # Multiply by the mask of values which have  #
            # saturated, ie these parameters will not be #
            # updated in future iterations.              #
            delta_j = delta_j * tmp_mask
        else:
            if n_layer == -1:
                in_layer = X_train
            else:
                in_layer = layer_out[n_layer]
            
            b_layer = nnet[bias_name]
            w_layer = nnet[weight_name]
            delta_j = np.matmul(
                delta_l, np.transpose(w_layer, axes=(1, 0)))
            if nnet[activation] == "relu":
                delta_j = delta_j * relu_derivative(in_layer)
            else:
                delta_j = delta_j * sigmoid_derivative(in_layer)
            
            # Update rules. #
            delta_b = np.mean(delta_l, axis=0)
            delta_w = np.mean(np.multiply(
                np.expand_dims(in_layer, axis=2), 
                np.expand_dims(delta_l, axis=1)), axis=0)
            
            # L2 Regularisation. #
            delta_b += lambda_reg / 2.0 * b_layer
            delta_w += lambda_reg / 2.0 * w_layer
            
            # Update the parameters. #
            nnet[bias_name] = nnet[bias_name] - learn_rate*delta_b
            nnet[weight_name] = nnet[weight_name] - learn_rate*delta_w
        
        # Update the delta. #
        delta_l = delta_j
    return nnet

# Neural Network Parameters. #
use_bias = True
in_dim   = 64
hid_dim  = [128, 256, 128]
out_dim  = 1
act_func = ["relu"] * len(hid_dim)
act_func = act_func + ["sigmoid"]
n_layers = len(hid_dim) + 1

nnet = {}
for n_layer in range(n_layers):
    activation  = "activation_" + str(n_layer)
    bias_name   = "bias_" + str(n_layer)
    weight_name = "layer_" + str(n_layer)
    
    if n_layer == 0:
        b_dim = hid_dim[n_layer]
        w_dim = (in_dim, hid_dim[n_layer])
    elif n_layer == (n_layers-1):
        b_dim = out_dim
        w_dim = (hid_dim[n_layer-1], out_dim)
    else:
        b_dim = hid_dim[n_layer]
        w_dim = (hid_dim[n_layer-1], hid_dim[n_layer])
    
    nnet[activation]  = act_func[n_layer]
    nnet[bias_name]   = np.zeros(shape=b_dim)
    nnet[weight_name] = \
        np.random.normal(scale=0.1, size=w_dim)

# Generate the data. #
n_data = 1000
X, y = make_classification(
    n_samples=n_data, n_features=in_dim, random_state=100)
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=42)

# Training Parameters. #
n_steps = 50
learn_rate = 0.10
lambda_reg = 0.01

y_test  = y_test.reshape((-1, 1))
y_train = y_train.reshape((-1, 1))

#X_input  = np.random.normal(size=(n_data, in_dim))
#y_output = np.random.uniform(size=(n_data, 1))
#y_output = np.where(y_output <= 0.5, 0.0, 1.0)

# Forward propagation. #
for n_step in range(n_steps):
    # Forward Pass. #
    layer_out = forward_pass(nnet, X_train)
    nnet_pred = layer_out[-1]
    
    # Binary Cross Entropy Loss. #
    ce_loss, tmp_mask = binary_ce(y_train, nnet_pred)
    print("Error:", str(ce_loss))
    
    n_valid = np.sum(tmp_mask)
    print("No. of Valid Points:", str(n_valid))
    
    if n_valid == 0:
        print("No more valid data to update parameters.")
        break
    
    # Update the parameters. #
    nnet = backward_pass(
        n_layers, nnet, X_train, y_train, nnet_pred, layer_out)
    
    # Print out the statistics. #
    tmp_out  = np.squeeze(y_train, axis=1)
    tmp_pred = np.squeeze(nnet_pred, axis=1)
    tmp_pred = np.where(tmp_pred <= 0.5, 0, 1)
    tmp_acc  = np.mean(np.where(tmp_out == tmp_pred, 1, 0))
    
    tmp_out  = np.squeeze(y_test, axis=1)
    tmp_pred = forward_pass(nnet, X_test)[-1]
    tmp_pred = np.squeeze(tmp_pred, axis=1)
    tmp_pred = np.where(tmp_pred <= 0.5, 0, 1)
    val_acc  = np.mean(np.where(tmp_out == tmp_pred, 1, 0))
    
    print("Train Accuracy at", str(n_step+1), 
          "step:", str(tmp_acc*100) + "%.")
    print("Validation Accuracy at", str(n_step+1), 
          "step:", str(val_acc*100) + "%.")
    print("-" * 50)

