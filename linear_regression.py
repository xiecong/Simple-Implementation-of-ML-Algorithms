import numpy as np
from sklearn.datasets import load_boston
# TO DO: improve performance and ridge regression


def squared_loss(y, pred):
	return np.square(pred - y).mean() / 2

def squared_loss_gradient(y, pred):
	return pred - y

class LinearRegression(object):
	def __init__(self):
		self.learning_rate = 1.5e-8
		self.embedding_dim = 1
		self.lmbda = 0.005 # regularization coefficient
		self.reg = 2


	def fit(self, x, y):
		n_dim = x.shape[1]
		self.b = 0
		self.w = np.random.randn(n_dim)

		for i in range(1000):
			grad = squared_loss_gradient(y, self.predict(x))
			self.b -= self.learning_rate * grad.sum()
			self.w -= self.learning_rate * grad.T.dot(x)
			self.regularization()
			if i % 10 == 0:
				print('loss {}'.format(squared_loss(self.predict(x), y)))

	def regularization(self):
		if(self.reg==1):
			self.w -= self.lmbda * np.sign(self.w)
			self.b -= self.lmbda * np.sign(self.b)
		elif(self.reg==2):
			self.w -= self.lmbda * self.w
			self.b -= self.lmbda * self.b

	def predict(self, x):
		return self.b + x.dot(self.w)

def main():
	data = load_boston()
	test_ratio = 0.2
	test_split = np.random.uniform(0, 1, len(data.data))
	train_x = data.data[test_split >= test_ratio]
	test_x = data.data[test_split < test_ratio]
	train_y = data.target[test_split >= test_ratio]
	test_y = data.target[test_split < test_ratio]

	lr = LinearRegression()
	lr.fit(train_x, train_y)
	print(squared_loss(lr.predict(train_x), train_y))
	print(squared_loss(lr.predict(test_x), test_y))


if __name__ == "__main__":
    main()