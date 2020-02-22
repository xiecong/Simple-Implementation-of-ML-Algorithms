import numpy as np
import requests
import re


def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def tanh(x):
	return np.tanh(x)

def dsigmoid(grad_a, act):
	return grad_a * (act - np.square(act))

def dtanh(grad_a, act):
	return grad_a * (1 - np.square(act))

def softmax(x):
	eps = 1e-20
	out = np.exp(x - np.max(x, axis=1).reshape(-1, 1))
	return out / (np.sum(out, axis=1).reshape(-1, 1) + eps)

def cross_entropy(pred, y):
	return -(np.multiply(y, np.log(pred + 1e-20))).sum()


class LSTM(object):
	def __init__(self, n_input, n_hidden, n_label, n_t):
		self.loss = cross_entropy
		self.n_hidden, self.n_label = n_hidden, n_label
		self.lr, self.batch_size, self.epochs = 1, 32, 200
		self.eps = 1e-20
		self.n_t = n_t
		
		self.w_f, self.w_i, self.w_c, self.w_o = [np.random.randn(n_input, self.n_hidden) / n_input for _ in range(4)]
		self.u_f, self.u_i, self.u_c, self.u_o = [np.random.randn(self.n_hidden, self.n_hidden) / self.n_hidden for _ in range(4)]
		self.b_f, self.b_i, self.b_c, self.b_o = [np.random.randn(1, self.n_hidden) for _ in range(4)]
		self.u_v, self.b_v = np.random.randn(self.n_hidden, self.n_label) / self.n_hidden, np.random.randn(1, self.n_label)

		self.param_list = [
			self.w_f, self.w_i, self.w_c, self.w_o, 
			self.u_f, self.u_i, self.u_c, self.u_o, self.u_v,
			self.b_f, self.b_i, self.b_c, self.b_o, self.b_v
		]
		self.mom_list = [np.zeros_like(param) for param in self.param_list]
		self.cache_list = [np.zeros_like(param) for param in self.param_list]

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
				h, f, i, c, o, c_bar, grad_f, grad_i, grad_o, grad_c, grad_c_bar = [
					np.zeros((n_t * b_size, self.n_hidden)) for _ in range(11)
				]
				
				# forward pass
				for t in range(n_t):
					t_idx = np.arange(t * b_size, (t + 1) * b_size)
					t_idx_prev = t_idx - b_size if t > 0 else t_idx

					xt_batch, ht_prev = x_batch[t_idx], h[t_idx_prev]

					f[t_idx] = sigmoid(xt_batch @ self.w_f + ht_prev @ self.u_f + self.b_f)
					i[t_idx] = sigmoid(xt_batch @ self.w_i + ht_prev @ self.u_i + self.b_i)
					o[t_idx] = sigmoid(xt_batch @ self.w_o + ht_prev @ self.u_o + self.b_o)
					c_bar[t_idx] = tanh(xt_batch @ self.w_c + ht_prev @ self.u_c + self.b_c)
					c[t_idx] = f[t_idx] * c[t_idx_prev] + i[t_idx] * c_bar[t_idx]
					h[t_idx] = o[t_idx] * tanh(c[t_idx])

				c_prev = np.zeros(c.shape)
				c_prev[b_size:, :] = c[:-b_size, :]
				h_prev = np.zeros(h.shape)
				h_prev[b_size:, :] = h[:-b_size, :]

				# back propagation through time
				grad_v = softmax(h @ self.u_v + self.b_v) - y_batch
				grad_h = grad_v @ self.u_v.T

				for t in reversed(range(0, n_t)):
					t_idx = np.arange(t * b_size, (t + 1) * b_size)
					t_idx_next = t_idx + b_size if t < n_t - 1 else t_idx
					grad_h[t_idx] += (
						dsigmoid(grad_f[t_idx_next], f[t_idx_next]) @ self.u_f.T +
						dsigmoid(grad_i[t_idx_next], i[t_idx_next]) @ self.u_i.T +
						dsigmoid(grad_o[t_idx_next], o[t_idx_next]) @ self.u_o.T +
						dtanh(grad_c_bar[t_idx_next], c_bar[t_idx_next]) @ self.u_c.T
					)
					grad_c[t_idx] = o[t_idx] * grad_h[t_idx] * (1 - np.square(np.tanh(c[t_idx]))) + f[t_idx_next] * grad_c[t_idx_next]
					grad_f[t_idx] = grad_c[t_idx] * c_prev[t_idx]
					grad_i[t_idx] = grad_c[t_idx] * c_bar[t_idx]
					grad_o[t_idx] = grad_h[t_idx] * tanh(c[t_idx])
					grad_c_bar[t_idx] = grad_c[t_idx] * i[t_idx]

				self.adam(
					grad_list=[
						x_batch.T @ dsigmoid(grad_f, f), x_batch.T @ dsigmoid(grad_i, i), x_batch.T @ dtanh(grad_c_bar, c_bar), x_batch.T @ dsigmoid(grad_o, o),
						h_prev.T @ dsigmoid(grad_f, f), h_prev.T @ dsigmoid(grad_i, i), h_prev.T @ dtanh(grad_c_bar, c_bar), h_prev.T @ dsigmoid(grad_o, o), h.T @ grad_v,
						constant @ dsigmoid(grad_f, f), constant @ dsigmoid(grad_i, i), constant @ dtanh(grad_c_bar, c_bar), constant @ dsigmoid(grad_o, o), constant @ grad_v
					]
				)
				self.regularization()
			if hasattr(self, 'ix_to_word'):
				print(self.sample(np.random.randint(n_input), np.random.randn(1, self.n_hidden), np.random.randn(1, self.n_hidden), n_t * 4))
			print(self.loss(self.predict(x).reshape(n_t * n_data, self.n_label), y.reshape(n_t * n_data, self.n_label)))

	def gradient_check(self, x, label):
		n_t, n_data, n_input = x.shape
		y = np.zeros((n_t * n_data, self.n_label))
		y[np.arange(n_t * n_data), label.flatten()] = 1
		x_batch = x.reshape(n_t * n_data, n_input)
		h, f, i, c, o, c_bar, grad_f, grad_i, grad_o, grad_c, grad_c_bar = [
			np.zeros((n_t * n_data, self.n_hidden)) for _ in range(11)
		]
		
		constant = np.ones((1, n_data*n_t))
		# forward pass
		for t in range(n_t):
			t_idx = np.arange(t * n_data, (t + 1) * n_data)
			t_idx_prev = t_idx - n_data if t > 0 else t_idx

			xt_batch, ht_prev = x_batch[t_idx], h[t_idx_prev]
			f[t_idx] = sigmoid(xt_batch @ self.w_f + ht_prev @ self.u_f + self.b_f)
			i[t_idx] = sigmoid(xt_batch @ self.w_i + ht_prev @ self.u_i + self.b_i)
			o[t_idx] = sigmoid(xt_batch @ self.w_o + ht_prev @ self.u_o + self.b_o)
			c_bar[t_idx] = tanh(xt_batch @ self.w_c + ht_prev @ self.u_c + self.b_c)
			c[t_idx] = f[t_idx] * c[t_idx_prev] + i[t_idx] * c_bar[t_idx]
			h[t_idx] = o[t_idx] * tanh(c[t_idx])

		c_prev = np.zeros(c.shape)
		c_prev[n_data:, :] = c[:-n_data, :]
		h_prev = np.zeros(h.shape)
		h_prev[n_data:, :] = h[:-n_data, :]

		# back propagation through time
		grad_v = softmax(h @ self.u_v + self.b_v) - y
		grad_h = grad_v @ self.u_v.T

		for t in reversed(range(0, n_t)):
			t_idx = np.arange(t * n_data, (t + 1) * n_data)
			t_idx_next = t_idx + n_data if t < n_t - 1 else t_idx
			grad_h[t_idx] += (
				dsigmoid(grad_f[t_idx_next], f[t_idx_next]) @ self.u_f.T +
				dsigmoid(grad_i[t_idx_next], i[t_idx_next]) @ self.u_i.T +
				dsigmoid(grad_o[t_idx_next], o[t_idx_next]) @ self.u_o.T +
				dtanh(grad_c_bar[t_idx_next], c_bar[t_idx_next]) @ self.u_c.T
			)
			grad_c[t_idx] = o[t_idx] * grad_h[t_idx] * (1 - np.square(np.tanh(c[t_idx]))) + f[t_idx_next] * grad_c[t_idx_next]
			grad_f[t_idx] = grad_c[t_idx] * c_prev[t_idx]
			grad_i[t_idx] = grad_c[t_idx] * c_bar[t_idx]
			grad_o[t_idx] = grad_h[t_idx] * tanh(c[t_idx])
			grad_c_bar[t_idx] = grad_c[t_idx] * i[t_idx]

		index = (0, 1)
		eps = 1e-4
		for j, grad in enumerate([
			x_batch.T @ dsigmoid(grad_f, f), x_batch.T @ dsigmoid(grad_i, i), x_batch.T @ dtanh(grad_c_bar, c_bar), x_batch.T @ dsigmoid(grad_o, o),
			h_prev.T @ dsigmoid(grad_f, f), h_prev.T @ dsigmoid(grad_i, i), h_prev.T @ dtanh(grad_c_bar, c_bar), h_prev.T @ dsigmoid(grad_o, o), h.T @ grad_v,
			constant @ dsigmoid(grad_f, f), constant @ dsigmoid(grad_i, i), constant @ dtanh(grad_c_bar, c_bar), constant @ dsigmoid(grad_o, o), constant @ grad_v
		]):
			preds = [0, 0]
			for sign in [+1, -1]:
				params = [param.copy() for param in self.param_list]
				params[j][index] += sign * eps

				w_f_a, w_i_a, w_c_a, w_o_a, u_f_a, u_i_a, u_c_a, u_o_a, u_v_a, b_f_a, b_i_a, b_c_a, b_o_a, b_v_a = params
				h_a, f_a, i_a, c_a, o_a, c_bar_a = [
					np.zeros((n_t * n_data, self.n_hidden)) for _ in range(6)
				]

				for t in range(n_t):
					t_idx = np.arange(t * n_data, (t + 1) * n_data)
					t_idx_prev = t_idx - n_data if t > 0 else t_idx

					xt_batch, ht_prev_a = x_batch[t_idx], h_a[t_idx_prev]
					f_a[t_idx] = sigmoid(xt_batch @ w_f_a + ht_prev_a @ u_f_a + b_f_a)
					i_a[t_idx] = sigmoid(xt_batch @ w_i_a + ht_prev_a @ u_i_a + b_i_a)
					o_a[t_idx] = sigmoid(xt_batch @ w_o_a + ht_prev_a @ u_o_a + b_o_a)
					c_bar_a[t_idx] = tanh(xt_batch @ w_c_a + ht_prev_a @ u_c_a + b_c_a)
					c_a[t_idx] = f_a[t_idx] * c_a[t_idx_prev] + i_a[t_idx] * c_bar_a[t_idx]
					h_a[t_idx] = o_a[t_idx] * tanh(c_a[t_idx])

				preds[(sign + 1) // 2] = cross_entropy(softmax(h_a @ u_v_a + b_v_a), y)
			print('gradient_check', j, ((preds[1] - preds[0]) / eps / 2 - grad[index])/eps/eps)


	def sgd(self, grad_list):
		alpha = self.lr / self.batch_size / self.n_t
		for params, grads in zip(self.param_list, grad_list): 
			params -= alpha * grads

	def adam(self, grad_list):
		beta1 = 0.9
		beta2 = 0.999
		alpha = self.lr / self.batch_size / self.n_t
		for params, grads, mom, cache in zip(
			self.param_list, grad_list, self.mom_list, self.cache_list
		):
			mom += (beta1 - 1) * mom + (1 - beta1) * grads
			cache += (beta2 - 1) * cache + (1 - beta2) * np.square(grads)
			params -= alpha * mom / (np.sqrt(cache) + self.eps)

	def regularization(self):
		lbd = 1e-5
		for params in self.param_list:
			params -= lbd * params

	def predict(self, x):
		n_t, n_data, n_input = x.shape
		h, f, i, c, o = [np.zeros((n_t * n_data, self.n_hidden)) for _ in range(5)]
		# forward pass
		for t in range(n_t):
			t_idx = np.arange(t * n_data, (t + 1) * n_data)
			t_idx_prev = t_idx - n_data if t > 0 else t_idx
			f[t_idx] = sigmoid(x[t] @ self.w_f + h[t_idx_prev] @ self.u_f + self.b_f)
			i[t_idx] = sigmoid(x[t] @ self.w_i + h[t_idx_prev] @ self.u_i + self.b_i)
			o[t_idx] = sigmoid(x[t] @ self.w_o + h[t_idx_prev] @ self.u_o + self.b_o)
			c[t_idx] = f[t_idx] * c[t_idx_prev] + i[t_idx] * tanh(x[t] @ self.w_c + h[t_idx_prev] @ self.u_c + self.b_c)
			h[t_idx] = o[t_idx] * tanh(c[t_idx])
		return softmax(h @ self.u_v + self.b_v).reshape(n_t, n_data, self.n_label)

	def sample(self, x_idx, h, c, seq_length):
		n_input = self.w_f.shape[0]
		seq = [x_idx]
		for t in range(seq_length):
			x = np.zeros((1, n_input))
			x[0, seq[-1]] = 1

			f = sigmoid(x @ self.w_f + h @ self.u_f + self.b_f)
			i = sigmoid(x @ self.w_i + h @ self.u_i + self.b_i)
			o = sigmoid(x @ self.w_o + h @ self.u_o + self.b_o)
			c = f * c + i * tanh(x @ self.w_c + h @ self.u_c + self.b_c)
			h = o * tanh(c)
			y = softmax(h @ self.u_v + self.b_v)
			seq.append(np.random.choice(range(n_input), p=y.flatten()))
		return ''.join(np.vectorize(self.ix_to_word.get)(np.array(seq)).tolist())


def text_generation(use_word=True):
	text = requests.get('http://www.gutenberg.org/cache/epub/11/pg11.txt').text
	if use_word:
		text = [word+' ' for word in re.sub("[^a-zA-Z]", " ", text).lower().split()]

	words = sorted(list(set(text)))
	text_size, vocab_size = len(text), len(words)

	print(f'text has {text_size} characters, {vocab_size} unique.')
	word_to_ix = {word:i for i, word in enumerate(words)}
	ix_to_word = {i:word for i, word in enumerate(words)}

	seq_length = 50
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

	lstm = LSTM(vocab_size, 500, vocab_size, seq_length)
	lstm.ix_to_word = ix_to_word
	lstm.gradient_check(train_x[:,np.arange(32),:], train_y[:,np.arange(32)])
	lstm.fit(train_x, train_y)
	print('train loss', (np.argmax(lstm.predict(train_x), axis=2)==train_y).sum()/(train_y.shape[0] * train_y.shape[1]))
	print('test loss', (np.argmax(lstm.predict(test_x), axis=2)==test_y).sum()/(test_y.shape[0] * test_y.shape[1]))


def main():
	text_generation(use_word=False)


if __name__ == "__main__":
	main()