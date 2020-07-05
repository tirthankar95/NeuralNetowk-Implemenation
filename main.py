from NeuralNetwork import NN
import numpy as np


o=NN(0.1,2000)
Xtr=np.array([[0,0],[0,1],[1,0],[1,1]]).T
Ytr=np.array([[0,1,1,0]])
layerInfo=[[2,'tanh'],[1,'sigmoid']]
o.train(Xtr,Ytr,layerInfo,'Log')
print('Log ->\n',o.test(Xtr,layerInfo))
o.train(Xtr,Ytr,layerInfo,'Quadratic')
print('Quadratic ->\n',o.test(Xtr,layerInfo))