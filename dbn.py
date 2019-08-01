import numpy as np
from sklearn.datasets import load_digits, fetch_openml


def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def dsigmoid(grad_a, act):
	return np.multiply(grad_a, act - np.square(act))

def softmax(x):
	eps = 1e-8
	out = np.exp(x - np.max(x, axis=1).reshape(-1, 1))
	return out / (np.sum(out, axis=1).reshape(-1, 1) + eps)


class RBM(object):
	def __init__(self, n_v, n_h, epochs=2000):
		self.w = np.random.randn(n_v, n_h)
		self.a = np.random.randn(1, n_v)
		self.b = np.random.randn(1, n_h)

		self.mom_w = np.zeros_like(self.w)
		self.mom_a = np.zeros_like(self.a)
		self.mom_b = np.zeros_like(self.b)

		self.lr = 0.05
		self.batch_size = 64
		self.max_epochs = epochs
		self.gamma = 0.5
		self.decay = 1 - 1e-4

	def fit(self, v):
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
				
				alpha = self.lr / self.batch_size
				self.mom_w = self.gamma * self.mom_w + alpha * (np.matmul(v1.T, p_h1) - np.matmul(v0.T, p_h0))
				self.w = (self.w - self.mom_w) * self.decay
				self.mom_a = self.gamma * self.mom_a + alpha * np.matmul(np.ones((1, self.batch_size)), v1 - v0)
				self.a = (self.a - self.mom_a) * self.decay
				self.mom_b = self.gamma * self.mom_b + alpha * np.matmul(np.ones((1, self.batch_size)), p_h1 - p_h0)
				self.b = (self.b - self.mom_b) * self.decay
			if j % 99 == 0: print(np.square(self.marginal_v(self.marginal_h(v)) - v).sum())
		# print(np.around(self.marginal_v(self.marginal_h(v)), 3))

	def marginal_v(self, h):
		return sigmoid(self.a + np.matmul(h, self.w.T))

	def marginal_h(self, v):
		return sigmoid(self.b + np.matmul(v, self.w))


class DBN(object):
	def __init__(self, layers):
		self.rbms = []
		for n_v, n_h in zip(layers[:-1], layers[1:]):
			self.rbms.append(RBM(n_v, n_h))
		self.epochs = 200
		self.batch_size = 32
		self.learning_rate = 0.01

	def pretrain(self, x):
		v = x
		for rbm in self.rbms:
			rbm.fit(v)
			v = rbm.marginal_h(v)

	def finetuning(self, x, labels):
		train_num = x.shape[0]
		y = np.zeros((train_num, self.n_labels))
		y[np.arange(train_num), labels] = 1
		l_num = len(self.rbms) + 1
		bvec = np.ones((1, self.batch_size))
		for epoch in range(self.epochs):
			permut=np.random.permutation(train_num//self.batch_size*self.batch_size).reshape(-1, self.batch_size)
			for b_idx in range(permut.shape[0]):
				act = [x[permut[b_idx,:]]]
				for rbm in self.rbms:
					act.append(rbm.marginal_h(act[-1]))
				pred = softmax(np.matmul(act[-1], self.bp_w) + self.bp_b)
				
				grad_a, grad_w, grad_b = [np.empty]*(l_num+1), [np.empty]*l_num, [np.empty]*l_num
				grad_a[l_num] = pred - y[permut[b_idx,:]]
				grad_w[l_num-1] = act[-1].T.dot(grad_a[l_num])
				grad_b[l_num-1] = bvec.dot(grad_a[l_num])

				for i in reversed(range(1, l_num)):
					grad_a[i] = grad_a[i+1].dot(self.rbms[i].w.T if i < len(self.rbms) else self.bp_w.T)
					grad_a[i] = dsigmoid(grad_a[i], act[i])
					grad_w[i-1] = act[i-1].T.dot(grad_a[i])
					grad_b[i-1] = bvec.dot(grad_a[i])

				self.sgd(grad_w, grad_b)
			print('epoch {}, loss: {}'.format(epoch, self.cross_entropy(self.predict(x), y)))

	def cross_entropy(self, pred, y):
		return -(np.multiply(y, np.log(pred + 1e-4))).mean()

	def sgd(self, grad_w, grad_b):
		l_num = len(self.rbms)
		alpha = self.learning_rate / self.batch_size
		for i in range(l_num):
			self.rbms[i].w -= alpha * grad_w[i]
			self.rbms[i].b -= alpha * grad_b[i]
		self.bp_w -= alpha * grad_w[l_num]
		self.bp_w -= alpha * grad_w[l_num]

	def fit(self, x, y):
		labels = np.unique(y)
		self.n_labels = len(labels)
		self.bp_w = np.random.randn(self.rbms[-1].b.shape[1], self.n_labels)
		self.bp_b = np.random.randn(1, self.n_labels)
		self.pretrain(x)
		self.finetuning(x, y)

	def predict(self, x):
		v = x
		for rbm in self.rbms:
			v = rbm.marginal_h(v)
		return softmax(np.matmul(v, self.bp_w) + self.bp_b)


def main():
	data = load_digits()
	x, y = data.data, data.target
	# x, y = fetch_openml('mnist_784', return_X_y=True, data_home="data")
	test_ratio = 0.2
	test_split = np.random.uniform(0, 1, x.shape[0])
	train_x, test_x = x[test_split >= test_ratio] / x.max(), x[test_split < test_ratio] / x.max()
	train_y, test_y = y.astype(np.int_)[test_split >= test_ratio], y.astype(np.int_)[test_split < test_ratio]

	dbn = DBN([train_x.shape[1], 100, 50])
	dbn.fit(train_x, train_y)
	print(sum(np.argmax(dbn.predict(train_x), axis=1) == train_y)/train_y.shape[0])
	print(sum(np.argmax(dbn.predict(test_x), axis=1) == test_y)/test_y.shape[0])


if __name__ == "__main__":
	main()