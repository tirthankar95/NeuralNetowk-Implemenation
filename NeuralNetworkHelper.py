import numpy as np 

def sigmoid(Z):
	return 1/(1+np.exp(-Z))

def tanh(Z):
	return np.tanh(Z)

def dsigmoid(A):
	return A*(1-A)
