import numpy as np
from sklearn import datasets

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegression():
	def __init__(self, x, y):
		self.x = x
		self.label_num = len(np.unique(y))	
		self.y = np.zeros((x.shape[0], self.label_num))
		self.y[np.arange(x.shape[0]), y] = 1
		self.w = np.random.randn(self.x.shape[1], self.label_num)
		self.b = np.random.randn(1, self.label_num)

	def loss(self): #using cross entropy as loss function
		eps = 1e-8
		h = self.predict(self.x)
		return -(np.multiply(self.y, np.log(h+eps)) + np.multiply((1 - self.y), np.log(1 - h+eps))).mean()

	def fit(self):
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

def main():
	data = datasets.load_digits()
	x = data.data
	y = data.target
	lr = LogisticRegression(x, y)
	lr.fit()
	res = lr.predict(lr.x)
	print(sum(y[i]==np.argmax(o) for i, o in enumerate(res))/y.shape[0])

if __name__ == "__main__":
    main()