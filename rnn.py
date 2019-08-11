import numpy as np


def relu(x):
	return np.maximum(x, 0)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def tanh(x):
	return np.tanh(x)

def linear(x):
	return x

def drelu(grad_a, act):
	grad_a[act <= 0] = 0
	return grad_a

def dsigmoid(grad_a, act):
	return np.multiply(grad_a, act - np.square(act))

def dtanh(grad_a, act):
	return np.multiply(grad_a, 1 - np.square(act))

def dlinear(grad_a, act):
	return grad_a

def softmax(x):
    eps = 1e-20
    out = np.exp(x - np.max(x, axis=1).reshape(-1, 1))
    return out / (np.sum(out, axis=1).reshape(-1, 1) + eps)

def cross_entropy(pred, y):
    return -(np.multiply(y, np.log(pred + 1e-20))).mean()

def squared_error(pred, y):
    return np.square(pred - y).mean() / 2


class RNN(object):
	def __init__(self, n_input, n_hidden, n_label, n_t):
		self.act_func, self.dact_func = tanh, dtanh
		self.loss = cross_entropy
		self.n_hidden, self.n_label = n_hidden, n_label
		self.lr, self.batch_size, self.epochs = 0.1, 32, 100
		self.eps = 1e-20
		self.n_t = n_t
		self.u = np.random.randn(n_input, self.n_hidden) / n_input
		self.w = np.random.randn(self.n_hidden, self.n_hidden) / self.n_hidden
		self.b = np.random.randn(1, self.n_hidden)
		self.v = np.random.randn(self.n_hidden, n_label) / self.n_hidden
		self.c = np.random.randn(1, self.n_label)

		self.mom_u, self.cache_u = np.zeros_like(self.u), np.zeros_like(self.u)
		self.mom_v, self.cache_v = np.zeros_like(self.v), np.zeros_like(self.v)
		self.mom_w, self.cache_w = np.zeros_like(self.w), np.zeros_like(self.w)
		self.mom_b, self.cache_b = np.zeros_like(self.b), np.zeros_like(self.b)
		self.mom_c, self.cache_c = np.zeros_like(self.c), np.zeros_like(self.c)

	def fit(self, x, label):
		b_size = self.batch_size
		n_t, n_data, n_input = x.shape
		y = np.zeros((n_t * n_data, self.n_label))
		y[np.arange(n_t * n_data), label.flatten()] = 1
		y = y.reshape((n_t, n_data, self.n_label))
		constant = np.ones((1, self.batch_size*n_t))

		for epoch in range(self.epochs):
			permut=np.random.permutation(n_data//b_size*b_size).reshape(-1, b_size)
			for b_idx in range(permut.shape[0]):
				x_batch = x[:, permut[b_idx, :]].reshape(n_t * b_size, n_input)
				y_batch = y[:, permut[b_idx, :]].reshape(n_t * b_size, self.n_label)
				h = np.zeros((n_t * b_size, self.n_hidden))

				for t in range(n_t):
					t_idx = np.arange(t * b_size, (t + 1) * b_size)
					t_idx_1 = t_idx - b_size if t > 0 else t_idx
					h[t_idx] = self.act_func(x_batch[t_idx].dot(self.u) + h[t_idx_1].dot(self.w) + self.b)

				grad_pred = softmax(h.dot(self.v) + self.c) - y_batch

				grad_h = grad_pred.dot(self.v.T)
				for t in reversed(range(1, n_t)):
					t_idx = np.arange(t * b_size, (t + 1) * b_size) 
					grad_h[t_idx - b_size] += self.dact_func(grad_h[t_idx], h[t_idx]).dot(self.w.T)

				grad_o = self.dact_func(grad_h, h)

				grad_w = h[:-b_size].T.dot(grad_o[b_size:])			
				grad_u = x_batch.T.dot(grad_o)
				grad_b = constant.dot(grad_o)

				grad_v = h.T.dot(grad_pred)
				grad_c = constant.dot(grad_pred)

				self.adam(grad_u=grad_u, grad_w=grad_w, grad_b=grad_b, grad_v=grad_v, grad_c=grad_c)
				self.regularization()
			print(self.loss(self.predict(x).reshape(n_t * n_data, self.n_label), y.reshape(n_t * n_data, self.n_label)))

	def sgd(self, grad_u, grad_w, grad_b, grad_v, grad_c):
		alpha = self.lr / self.batch_size / self.n_t
		for params, grads in zip([self.u, self.w, self.b, self.v, self.c], [grad_u, grad_w, grad_b, grad_v, grad_c]): 
			params -= alpha * grads

	def sgd(self, grad_u, grad_w, grad_b, grad_v, grad_c):
		alpha = self.lr / self.batch_size / self.n_t
		for params, grads in zip([self.u, self.w, self.b, self.v, self.c], [grad_u, grad_w, grad_b, grad_v, grad_c]): 
			params -= alpha * grads

	def adam(self, grad_u, grad_w, grad_b, grad_v, grad_c):
		beta1 = 0.9
		beta2 = 0.999
		eps = 1e-20
		alpha = self.lr / self.batch_size / self.n_t
		for params, grads, mom, cache in zip(
			[self.u, self.w, self.b, self.v, self.c],
			[grad_u, grad_w, grad_b, grad_v, grad_c],
			[self.mom_u, self.mom_w, self.mom_b, self.mom_v, self.mom_c],
			[self.cache_u, self.cache_w, self.cache_b, self.cache_v, self.cache_c]
		):
			mom = beta1 * mom + (1 - beta1) * grads
			cache = beta2 * cache + (1 - beta2) * np.square(grads)
			params -= alpha * mom / (np.sqrt(cache) + eps)

	def regularization(self):
		lbd = 1e-4
		for params in [self.u, self.w, self.b, self.v, self.c]:
			params -= lbd * params

	def predict(self, x):
		n_t, n_data, n_input = x.shape
		h = np.zeros((n_t * n_data, self.n_hidden))
		for t in range(n_t):
			t_idx = np.arange(t * n_data, (t + 1) * n_data)
			t_idx_1 = t_idx - n_data if t > 0 else t_idx
			h[t_idx] = self.act_func(x[t].dot(self.u) + h[t_idx_1].dot(self.w) + self.b)
		return softmax(h.dot(self.v) + self.c).reshape(n_t, n_data, self.n_label)

def binary_add_test():
	binary_dim = 8
	max_num = pow(2, binary_dim)
	binary = np.flip(np.unpackbits(np.array([range(max_num)],dtype=np.uint8).T,axis=1),axis=1)
	numbers = np.random.randint(max_num/2, size=(8192, 2))
	x, y = binary[numbers].transpose(2,0,1), binary[numbers.sum(axis=1)].transpose()

	test_ratio = 0.2
	test_split = np.random.uniform(0, 1, numbers.shape[0])
	train_x, test_x = x[:,test_split >= test_ratio,:] , x[:,test_split < test_ratio,:]
	train_y, test_y = y[:,test_split >= test_ratio], y[:,test_split < test_ratio]

	rnn = RNN(2, 3, 2, binary_dim)
	rnn.fit(train_x, train_y)
	print((np.argmax(rnn.predict(train_x), axis=2)==train_y).sum()/(train_y.shape[0] * train_y.shape[1]))
	print((np.argmax(rnn.predict(test_x), axis=2)==test_y).sum()/(test_y.shape[0] * test_y.shape[1]))


def main():
	binary_add_test()

if __name__ == "__main__":
	main()