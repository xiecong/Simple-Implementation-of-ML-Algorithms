import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegression():
	def __init__(self):
		self.x = np.loadtxt("data/iris.data", delimiter = ',', usecols = (0,1,2,3), dtype = float)
		y = np.loadtxt("data/iris.data", delimiter = ',', usecols = (4), dtype = str).reshape(-1,1)
		class_mapping = {"b'Iris-setosa'":[1,0,0], "b'Iris-versicolor'":[0,1,0], "b'Iris-virginica'":[0,0,1]}
		self.y = np.array([class_mapping[yi[0]] for yi in y])
		self.w = np.random.randn(self.x.shape[1], 3)
		self.b = np.random.randn(1, 3)

	def loss(self): #using cross entropy as loss function
		eps = 1e-8
		h = self.predict(self.x)
		return -(np.multiply(self.y, np.log(h+eps)) + np.multiply((1 - self.y), np.log(1 - h+eps))).mean()

	def train(self):
		train_num = self.x.shape[0]
		learning_rate = 0.01
		for i in range(5000):
			h = sigmoid(self.x.dot(self.w)+self.b)
			g_w = self.x.T.dot(h-self.y) / train_num
			g_b = (h-self.y).sum() / train_num
			self.w -= learning_rate * g_w
			self.b -= learning_rate * g_b
			#print(lr.loss())

	def predict(self, x):
		return sigmoid(x.dot(self.w) + self.b)

lr = LogisticRegression()
lr.train()
print([np.argmax(o) for o in lr.predict(lr.x)])