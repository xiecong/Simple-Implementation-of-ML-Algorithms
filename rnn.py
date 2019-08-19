import numpy as np
import requests
import re
# TODO add sentence tokenizer


def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def tanh(x):
	return np.tanh(x)

def dsigmoid(grad_a, act):
	return np.multiply(grad_a, act - np.square(act))

def dtanh(grad_a, act):
	return np.multiply(grad_a, 1 - np.square(act))

def softmax(x):
	eps = 1e-20
	out = np.exp(x - np.max(x, axis=1).reshape(-1, 1))
	return out / (np.sum(out, axis=1).reshape(-1, 1) + eps)

def cross_entropy(pred, y):
	return -(np.multiply(y, np.log(pred + 1e-20))).sum()


class RNN(object):
	def __init__(self, n_input, n_hidden, n_label, n_t):
		self.act_func, self.dact_func = tanh, dtanh
		self.loss = cross_entropy
		self.n_hidden, self.n_label = n_hidden, n_label
		self.lr, self.batch_size, self.epochs = 0.5, 32, 200
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

				for grads in [grad_u, grad_w, grad_b, grad_v, grad_c]:
					np.clip(grads, -10, 10, out=grads)

				self.adam(grad_u=grad_u, grad_w=grad_w, grad_b=grad_b, grad_v=grad_v, grad_c=grad_c)
				self.regularization()
			if hasattr(self, 'ix_to_word'):
				print(self.sample(np.random.randint(n_input), np.random.randn(1, self.n_hidden), n_t * 4))
			print(self.loss(self.predict(x).reshape(n_t * n_data, self.n_label), y.reshape(n_t * n_data, self.n_label)))

	def gradient_check(self, x, label):
		n_t, n_data, n_input = x.shape
		y = np.zeros((n_t * n_data, self.n_label))
		y[np.arange(n_t * n_data), label.flatten()] = 1
		x_batch = x.reshape(n_t * n_data, n_input)
		h = np.zeros((n_t * n_data, self.n_hidden))

		for t in range(n_t):
			t_idx = np.arange(t * n_data, (t + 1) * n_data)
			t_idx_1 = t_idx - n_data if t > 0 else t_idx
			h[t_idx] = self.act_func(x_batch[t_idx].dot(self.u) + h[t_idx_1].dot(self.w) + self.b)
		grad_pred = softmax(h.dot(self.v) + self.c) - y
		grad_h = grad_pred.dot(self.v.T)
		for t in reversed(range(1, n_t)):
			t_idx = np.arange(t * n_data, (t + 1) * n_data) 
			grad_h[t_idx - n_data] += self.dact_func(grad_h[t_idx], h[t_idx]).dot(self.w.T)
		grad_o = self.dact_func(grad_h, h)
		
		index = (0, 1)
		dw = h[:-n_data].T.dot(grad_o[n_data:])[index]
		du = x_batch.T.dot(grad_o)[index]
		db = np.ones((1, n_data*n_t)).dot(grad_o)[index]
		dv = h.T.dot(grad_pred)[index]
		dc = np.ones((1, n_data*n_t)).dot(grad_pred)[index]

		eps = 1e-4
		for i, grad in enumerate([du, dv, dw, db, dc]):
			params = [
				self.u.copy(), self.u.copy(), self.v.copy(), self.v.copy(), self.w.copy(), self.w.copy(),
				self.b.copy(), self.b.copy(), self.c.copy(), self.c.copy()
			]
			params[2 * i + 0][index]+=eps
			params[2 * i + 1][index]-=eps
			h_1, h_2 = np.zeros((n_t * n_data, self.n_hidden)), np.zeros((n_t * n_data, self.n_hidden))
			for t in range(n_t):
				t_idx = np.arange(t * n_data, (t + 1) * n_data)
				t_idx_1 = t_idx - n_data if t > 0 else t_idx
				h_1[t_idx] = self.act_func(x_batch[t_idx].dot(params[0]) + h_1[t_idx_1].dot(params[4]) + params[6])
				h_2[t_idx] = self.act_func(x_batch[t_idx].dot(params[1]) + h_2[t_idx_1].dot(params[5]) + params[7])
			pred_1 = cross_entropy(softmax(h_1.dot(params[2]) + params[8]), y)
			pred_2 = cross_entropy(softmax(h_2.dot(params[3]) + params[9]), y)
			print('gradient_check', ((pred_1 - pred_2) / eps / 2 - grad)/eps/eps)

	def sgd(self, grad_u, grad_w, grad_b, grad_v, grad_c):
		alpha = self.lr / self.batch_size / self.n_t
		for params, grads in zip([self.u, self.w, self.b, self.v, self.c], [grad_u, grad_w, grad_b, grad_v, grad_c]): 
			params -= alpha * grads

	def adam(self, grad_u, grad_w, grad_b, grad_v, grad_c):
		beta1 = 0.9
		beta2 = 0.999
		alpha = self.lr / self.batch_size / self.n_t
		for params, grads, mom, cache in zip(
			[self.u, self.w, self.b, self.v, self.c],
			[grad_u, grad_w, grad_b, grad_v, grad_c],
			[self.mom_u, self.mom_w, self.mom_b, self.mom_v, self.mom_c],
			[self.cache_u, self.cache_w, self.cache_b, self.cache_v, self.cache_c]
		):
			mom = beta1 * mom + (1 - beta1) * grads
			cache = beta2 * cache + (1 - beta2) * np.square(grads)
			params -= alpha * mom / (np.sqrt(cache) + self.eps)

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

	def sample(self, x_idx, h, seq_length):
		n_input = self.u.shape[0]
		seq = [x_idx]
		for t in range(seq_length):
			x = np.zeros((1, n_input))
			x[0, seq[-1]] = 1
			h = self.act_func(x.dot(self.u) + h.dot(self.w) + self.b)
			y = softmax(h.dot(self.v) + self.c)
			seq.append(np.random.choice(range(n_input), p=y.flatten()))
		return ''.join(np.vectorize(self.ix_to_word.get)(np.array(seq)).tolist())


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
	# rnn.gradient_check(train_x[:,np.arange(32),:], train_y[:,np.arange(32)])
	print('train loss', (np.argmax(rnn.predict(train_x), axis=2)==train_y).sum()/(train_y.shape[0] * train_y.shape[1]))
	print('test loss', (np.argmax(rnn.predict(test_x), axis=2)==test_y).sum()/(test_y.shape[0] * test_y.shape[1]))

def text_generation(use_word=True):
	text = requests.get('http://www.gutenberg.org/cache/epub/11/pg11.txt').text
	if use_word:
		text = [word+' ' for word in re.sub("[^a-zA-Z]", " ", text).lower().split()]

	words = sorted(list(set(text)))
	text_size, vocab_size = len(text), len(words)

	print(f'text has {text_size} characters, {vocab_size} unique.')
	word_to_ix = {word:i for i, word in enumerate(words)}
	ix_to_word = {i:word for i, word in enumerate(words)}

	seq_length = 25
	indices = np.vectorize(word_to_ix.get)(np.array(list(text)))
	data = np.zeros((text_size, vocab_size))
	data[np.arange(text_size), indices] = 1
	n_text = (text_size - 1) // seq_length
	x = data[:n_text * seq_length].reshape(n_text, seq_length, vocab_size).transpose(1,0,2)
	y = indices[1: n_text * seq_length + 1].reshape(n_text, seq_length).T

	test_ratio = 0.2
	test_split = np.random.uniform(0, 1, x.shape[1])
	train_x, test_x = x[:,test_split >= test_ratio,:] , x[:,test_split < test_ratio,:]
	train_y, test_y = y[:,test_split >= test_ratio], y[:,test_split < test_ratio]

	rnn = RNN(vocab_size, 500, vocab_size, seq_length)
	rnn.ix_to_word = ix_to_word
	rnn.fit(train_x, train_y)
	# rnn.gradient_check(train_x[:,np.arange(32),:], train_y[:,np.arange(32)])
	print('train loss', (np.argmax(rnn.predict(train_x), axis=2)==train_y).sum()/(train_y.shape[0] * train_y.shape[1]))
	print('test loss', (np.argmax(rnn.predict(test_x), axis=2)==test_y).sum()/(test_y.shape[0] * test_y.shape[1]))

def main():
	text_generation(use_word=False)
	# binary_add_test()

if __name__ == "__main__":
	main()