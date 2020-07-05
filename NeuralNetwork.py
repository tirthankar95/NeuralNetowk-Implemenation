import numpy as np
import matplotlib.pyplot as plt

class NN:
	def __init__(self,alpha=0.5,mxIter=1000):
		self.W=[]
		self.b=[]
		self.A=[]
		self.alpha=alpha
		self.mxIter=mxIter
	def sigmoid(self,Z):
		return 1/(1+np.exp(-Z))
	def help(self):
		print('Supported Log Functions -> 1.Quadratic 2.Log')
		print('Activation Functions -> 1.Sigmoid 2.Tanh 3.Relu')
# dimension of Xtr is (n,m); where n is the NoOf dimension and m is the NoOf of training example.
# dimension of Ytr is (k,m); where k is the NoOf output layer and m is the NoOf of training example.
	def train(self,Xtr,Ytr,layerInfo,lossFunc):
		graph=[]
		layers=len(layerInfo)
		n,m=Xtr.shape
		for i in range(layers):
			if i==0:
				Wtmp=np.random.rand(layerInfo[i][0],n)
				btmp=np.random.rand(layerInfo[i][0],1)
			else:
				Wtmp=np.random.rand(layerInfo[i][0],layerInfo[i-1][0])
				btmp=np.random.rand(layerInfo[i][0],1)
			self.W.append(Wtmp)
			self.b.append(btmp)
		for j in range(self.mxIter):
			for i in range(layers):
				if layerInfo[i][1]=='sigmoid':
					if i==0:
						Atmp=self.sigmoid(np.dot(self.W[i],Xtr)+self.b[i])
					else:
						Atmp=self.sigmoid(np.dot(self.W[i],self.A[i-1])+self.b[i])
				if layerInfo[i][1]=='tanh':
					if i==0:
						Atmp=np.tanh(np.dot(self.W[i],Xtr)+self.b[i])
					else:
						Atmp=np.tanh(np.dot(self.W[i],self.A[i-1])+self.b[i])
				self.A.append(Atmp);loss=0
			if lossFunc=='Quadratic':
				loss=(1/m)*( ((Ytr-self.A[layers-1])**2).sum() )
				dZ=2*(self.A[layers-1]-Ytr)*self.A[layers-1]*(1-self.A[layers-1])
			if lossFunc=='Log':
				loss=-(1/m)*( (Ytr*np.log(self.A[layers-1])+(1-Ytr)*np.log(1-self.A[layers-1])).sum() )
				dZ=self.A[layers-1]-Ytr
			if j%10==0:
				graph.append(loss)                
			for i in range(layers-1,-1,-1):
				if i!=0:
					# i-th layer (dW,dB,dA) || i-1th layer (dZ)
					dW=(1/m)*np.dot(dZ,self.A[i-1].T)
					dB=(1/m)*np.sum(dZ,axis=1,keepdims=True)
					dA=np.dot(self.W[i].T,dZ)
					dZ=np.multiply(dA,self.A[i-1]*(1-self.A[i-1]))
				else:
					dW=(1/m)*np.dot(dZ,Xtr.T)
					dB=(1/m)*np.sum(dZ,axis=1,keepdims=True)                    
				self.W[i]=self.W[i]-self.alpha*dW
				self.b[i]=self.b[i]-self.alpha*dB
			self.A=[]
		x=[i*10 for i in range(len(graph))]
		plt.plot(x,graph)
		plt.xlabel('Iterations ->')
		plt.ylabel('Loss ->')
		plt.show()
	def test(self,X,layerInfo):
		Atmp=[]
		layers=len(layerInfo)
		for i in range(layers):
			if layerInfo[i][1]=='sigmoid':
				if i==0:
					Atmp=self.sigmoid(np.dot(self.W[i],X)+self.b[i])
				else:
					Atmp=self.sigmoid(np.dot(self.W[i],Atmp)+self.b[i])
		return Atmp




o=NN(0.5,5000)
Xtr=np.array([[0,0],[0,1],[1,0],[1,1]]).T
Ytr=np.array([[0,1,1,0]])
layerInfo=[[4,'sigmoid'],[2,'sigmoid'],[1,'sigmoid']]
o.train(Xtr,Ytr,layerInfo,'Log')
print('Log ->\n',o.test(Xtr,layerInfo))
o.train(Xtr,Ytr,layerInfo,'Quadratic')
print('Quadratic ->\n',o.test(Xtr,layerInfo))