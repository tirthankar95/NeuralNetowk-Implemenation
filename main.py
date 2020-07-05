from NeuralNetwork import NN
import numpy as np


o=NN(0.5,5000)
Xtr=np.array([[0,0],[0,1],[1,0],[1,1]]).T
Ytr=np.array([[0,1,1,0]])
layerInfo=[[4,'sigmoid'],[2,'sigmoid'],[1,'sigmoid']]
o.train(Xtr,Ytr,layerInfo,'Log')
print('Log ->\n',o.test(Xtr,layerInfo))
o.train(Xtr,Ytr,layerInfo,'Quadratic')
print('Quadratic ->\n',o.test(Xtr,layerInfo))