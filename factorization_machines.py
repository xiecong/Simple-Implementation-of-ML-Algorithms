import numpy as np
from sklearn.datasets import load_boston
# TO DO: improve performance and classification


def squared_loss(y, pred):
	return np.square(pred - y).mean() / 2

def squared_loss_gradient(y, pred):
	return pred - y

class FactorizationMachines(object):
	def __init__(self):
		self.learning_rate = 1e-14
		self.embedding_dim = 1
		self.lmbda = 0.005 # regularization coefficient
		self.reg = 2

	def fit(self, x, y):
		n_data = x.shape[0]
		n_dim = x.shape[1]
		self.w0 = 0
		self.w = np.random.randn(n_dim)
		self.v = np.random.randn(self.embedding_dim, n_dim)

		for i in range(1000):
			grad = squared_loss_gradient(y, self.predict(x))
			self.w0 -= self.learning_rate * grad.sum()
			self.w -= self.learning_rate * grad.T.dot(x)

			squares = np.repeat(np.square(x), self.embedding_dim, axis=0).reshape(n_data, -1, n_dim)
			vx = [np.matmul(vkx.reshape(-1,1), xi.reshape(1,-1)) for vkx, xi in zip(x.dot(self.v.T), x)]

			grad_v = np.array(vx) - squares
			self.v -= self.learning_rate * grad.T.dot(grad_v.reshape(n_data, -1)).reshape(self.v.shape)
			self.regularization()
			if i % 10 == 0:
				print('loss {}'.format(squared_loss(self.predict(x), y)))

	def regularization(self):
		if(self.reg==1):
			self.w0 -= self.lmbda * np.sign(self.w0)
			self.w -= self.lmbda * np.sign(self.w)
			self.v -= self.lmbda * np.sign(self.v)
		elif(self.reg==2):
			self.w0 -= self.lmbda * self.w0
			self.w -= self.lmbda * self.w
			self.v -= self.lmbda * self.v

	def predict(self, x):
		xvt = np.matmul(x, self.v.T)
		return self.w0 + x.dot(self.w) + (np.square(xvt).sum(axis=1) - np.square(x).dot(np.square(self.v).sum(axis=0))) / 2


def main():
	data = load_boston()
	test_ratio = 0.2
	test_split = np.random.uniform(0, 1, len(data.data))
	train_x = data.data[test_split >= test_ratio]
	test_x = data.data[test_split < test_ratio]
	train_y = data.target[test_split >= test_ratio]
	test_y = data.target[test_split < test_ratio]

	fm = FactorizationMachines()
	fm.fit(train_x, train_y)
	print(squared_loss(fm.predict(train_x), train_y))
	print(squared_loss(fm.predict(test_x), test_y))


if __name__ == "__main__":
    main()