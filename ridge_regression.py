import numpy as np
from numpy.linalg import inv
from sklearn.datasets import load_boston


def squared_loss(y, pred):
	return np.square(pred - y).mean() / 2

def squared_loss_gradient(y, pred):
	return pred - y


class RidgeRegression(object):
	def __init__(self):
		self.learning_rate = 0.1
		self.embedding_dim = 1
		self.lmbda = 0.001  # regularization coefficient
		self.reg = 2
		self.eps = 1e-12
		self.optimization = False

	def fit(self, x, y):
		if self.optimization:
			self.optimize(x, y)
		else:
			self.matrix_solver(x, y)

	def optimize(self, x, y):
		n_dim = x.shape[1]
		self.b = 0
		self.w = np.random.randn(n_dim)
		self.mom_w, self.cache_w = np.zeros(n_dim), np.zeros(n_dim)
		self.mom_b, self.cache_b = 0, 0

		for i in range(5000):
			grad = squared_loss_gradient(y, self.predict(x))
			self.adam(grad.T.dot(x), grad.sum())
			self.regularization()
			if i % 100 == 0:
				print('loss {}'.format(squared_loss(self.predict(x), y)))

	def matrix_solver(self, x, y):
		n_dim = x.shape[1]
		ext_x = np.c_[x, np.ones((x.shape[0], 1))]
		inv_matrix = inv(np.matmul(ext_x.T, ext_x) + self.lmbda * np.identity(n_dim + 1))
		ext_w = np.matmul(np.matmul(inv_matrix, ext_x.T), y.reshape(-1, 1))
		self.w = ext_w[:-1].flatten()
		self.b = ext_w[-1]

	def sgd(self, grad_w, grad_b):  # use a very small learning rate for sgd, e.g., 1e-8
		self.w -= self.learning_rate *grad_w
		self.b -= self.learning_rate * grad_b

	def adam(self, grad_w, grad_b):
		beta1 = 0.9
		beta2 = 0.999
		alpha = self.learning_rate
		self.mom_w = beta1 * self.mom_w + (1 - beta1) * grad_w
		self.cache_w = beta2 * self.cache_w + (1 - beta2) * np.square(grad_w)
		self.w -= alpha * self.mom_w / (np.sqrt(self.cache_w) + self.eps)
		self.mom_b = beta1 * self.mom_b + (1 - beta1) * grad_b
		self.cache_b = beta2 * self.cache_b + (1 - beta2) * np.square(grad_b)
		self.b -= alpha * self.mom_b / (np.sqrt(self.cache_b) + self.eps)

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

	rr = RidgeRegression()
	rr.fit(train_x, train_y)
	print(squared_loss(rr.predict(train_x), train_y))
	print(squared_loss(rr.predict(test_x), test_y))


if __name__ == "__main__":
    main()
