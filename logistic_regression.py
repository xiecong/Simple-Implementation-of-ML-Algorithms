import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegression():
	def __init__(self):
		self.x = np.loadtxt("data/iris.data", delimiter = ',', usecols = (0,1,2,3), dtype = float)
		y = np.loadtxt("data/iris.data", delimiter = ',', usecols = (4), dtype = str).reshape(-1,1)
		self.y = (y == "b'Iris-setosa'") * 1 # set to to a binary classification
		self.w = np.random.randn(self.x.shape[1], 1)
		self.b = 0

	def loss(self): #using cross entropy as loss function
		eps = 1e-8
		h = self.predict(self.x)
		return (-self.y * np.log(h+eps) - (1 - self.y) * np.log(1 - h+eps)).mean()

	def train(self):
		train_num = self.x.shape[0]
		learning_rate = 0.01
		for i in range(200):
			h = sigmoid(self.x.dot(self.w)+self.b)
			g_w = self.x.T.dot(h-self.y) / train_num
			g_b = (h-self.y).sum() / train_num
			self.w -= learning_rate * g_w
			self.b -= learning_rate * g_b
			print(self.loss())

	def predict(self, x):
		return sigmoid(x.dot(self.w) + self.b)>0.5

lr = LogisticRegression()
lr.train()
print(lr.predict(np.array([[4.9,3.0,1.4,0.2], [6.1,2.8,4.7,1.2]])))