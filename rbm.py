import numpy as np
from sklearn.datasets import load_digits, fetch_openml
import matplotlib.pyplot as plt


def sigmoid(x):
	return 1 / (1 + np.exp(-x))


class RBM(object):
	def __init__(self, n_v, n_h, epochs=200, lr=0.2):
		self.w = np.random.randn(n_v, n_h)
		self.a = np.random.randn(1, n_v)
		self.b = np.random.randn(1, n_h)

		self.mom_w, self.cache_w = np.zeros_like(self.w), np.zeros_like(self.w)
		self.mom_a, self.cache_a = np.zeros_like(self.a), np.zeros_like(self.a)
		self.mom_b, self.cache_b = np.zeros_like(self.b), np.zeros_like(self.b)

		self.lr = lr
		self.batch_size = 16
		self.max_epochs = epochs
		self.gamma = 0.5
		self.decay = 1 - 1e-4

	def fit(self, v):
		beta1 = 0.9
		beta2 = 0.999
		eps = 1e-20

		train_num = v.shape[0]
		for j in range(self.max_epochs):
			permut=np.random.permutation(train_num//self.batch_size*self.batch_size).reshape(-1, self.batch_size)
			for i in range(permut.shape[0]):
				v0 = v[permut[i],:]
				p_h0 = self.marginal_h(v0)
				h0 = 1 * (p_h0 >= np.random.uniform(0, 1, (self.batch_size, self.b.shape[1])))
				v1 = self.marginal_v(h0)
				p_h1 = self.marginal_h(v1)
				h1 = 1 * (p_h1 >= np.random.uniform(0, 1, (self.batch_size, self.b.shape[1])))

				grad_w = np.matmul(v1.T, p_h1) - np.matmul(v0.T, p_h0)
				grad_b = np.matmul(np.ones((1, self.batch_size)), p_h1 - p_h0)
				grad_a = np.matmul(np.ones((1, self.batch_size)), v1 - v0)

				alpha = self.lr / self.batch_size
				mom_scaler = 1 - beta1 ** (j + 1)
				cache_scaler = 1 - beta2 ** (j + 1)

				self.mom_w = beta1 * self.mom_w + (1 - beta1) * grad_w
				self.cache_w = beta2 * self.cache_w + (1 - beta2) * np.square(grad_w)
				self.w -= alpha * self.mom_w / mom_scaler / (np.sqrt(self.cache_w / cache_scaler) + eps)
				self.mom_b = beta1 * self.mom_b + (1 - beta1) * grad_b
				self.cache_b = beta2 * self.cache_b + (1 - beta2) * np.square(grad_b)
				self.b -= alpha * self.mom_b / mom_scaler / (np.sqrt(self.cache_b / cache_scaler) + eps)
				self.mom_a = beta1 * self.mom_a + (1 - beta1) * grad_a
				self.cache_a = beta2 * self.cache_a + (1 - beta2) * np.square(grad_a)
				self.a -= alpha * self.mom_a / mom_scaler / (np.sqrt(self.cache_a / cache_scaler) + eps)

				self.w *= self.decay
				self.a *= self.decay
				self.b *= self.decay
			if j % 9 == 0:
				print('squared loss', np.square(self.marginal_v(self.marginal_h(v)) - v).sum())
		# print(np.around(self.marginal_v(self.marginal_h(v)), 3))

	def marginal_v(self, h):
		return sigmoid(self.a + np.matmul(h, self.w.T))

	def marginal_h(self, v):
		return sigmoid(self.b + np.matmul(v, self.w))

	
def main():
	data = load_digits() # fetch_openml('mnist_784', return_X_y=True, data_home="data")
	x, y = data.data, data.target
	test_ratio = 0.2
	test_split = np.random.uniform(0, 1, x.shape[0])
	train_x, test_x = x[test_split >= test_ratio] / x.max(), x[test_split < test_ratio] / x.max()
	train_y, test_y = y.astype(np.int_)[test_split >= test_ratio], y.astype(np.int_)[test_split < test_ratio]

	rbm = RBM(x.shape[1], 20)
	rbm.fit(train_x)
	print(np.square(rbm.marginal_v(rbm.marginal_h(train_x)) - train_x).sum())
	print(np.square(rbm.marginal_v(rbm.marginal_h(test_x)) - test_x).sum())

	for i in range(10):
		plt.subplot(2, 10, i+1)
		plt.imshow(test_x[test_y == i].mean(axis=0).reshape(8, 8), cmap='gray', vmin=0, vmax=1)
		plt.subplot(2, 10, i+11)
		plt.imshow(rbm.marginal_v(rbm.marginal_h(test_x[test_y == i])).mean(axis=0).reshape(8, 8), cmap='gray', vmin=0, vmax=1)
	plt.show()


if __name__ == "__main__":
	main()
