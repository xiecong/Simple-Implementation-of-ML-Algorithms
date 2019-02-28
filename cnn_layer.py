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

class Layer(object):
	def __init__(self, has_param):
		self.act_funcs = {'ReLU': relu, 'Sigmoid': sigmoid, 'Tanh': tanh, "Linear": linear}
		self.dact_funcs = {'ReLU': drelu, 'Sigmoid': dsigmoid, 'Tanh': dtanh, "Linear": dlinear}
		self.gradient_funcs = {'Adam':self.adam, "SGD": self.sgd}
		self.learning_rate = 1e-2
		self.weight_decay = 1e-4
		self.has_param = has_param

	def forward(self, x):
		pass

	def gradient(self, grad_act):
		pass

	def backward(self, grad_type):
		if self.has_param:
			self.regularize()
			self.gradient_funcs[grad_type]()

	def regularize(self):
		self.w *= (1 - self.weight_decay)
		self.b *= (1 - self.weight_decay)

	def adam(self):
		beta1 = 0.9
		beta2 = 0.999
		eps = 1e-8
		alpha = self.learning_rate
		self.mom_w = beta1 * self.mom_w + (1 - beta1) * self.grad_w
		self.cache_w = beta2 * self.cache_w + (1 - beta2) * np.square(self.grad_w)
		self.w -= alpha * self.mom_w / (np.sqrt(self.cache_w) + eps)
		self.mom_b = beta1 * self.mom_b + (1 - beta1) * self.grad_b
		self.cache_b = beta2 * self.cache_b + (1 - beta2) * np.square(self.grad_b)
		self.b -= alpha * self.mom_b / (np.sqrt(self.cache_b) + eps)

	def sgd(self):
		self.w -= self.learning_rate * self.grad_w
		self.b -= self.learning_rate * self.grad_b

class Conv(Layer):
	def __init__(self, in_shape, k_size, k_num, act_type, stride=1):
		super(Conv, self).__init__(has_param=True)
		self.act_func = self.act_funcs[act_type]
		self.dact_func = self.dact_funcs[act_type]
		self.in_shape = in_shape
		channel, height, width = in_shape
		self.k_size = k_size
		self.w = np.random.randn(channel*k_size*k_size, k_num)
		self.b = np.random.randn(1,k_num)

		self.mom_w = np.zeros_like(self.w)
		self.cache_w = np.zeros_like(self.w)
		self.mom_b = np.zeros_like(self.b)
		self.cache_b = np.zeros_like(self.b)

		self.out_shape = (k_num, (height-k_size+1)//stride, (width-k_size+1)//stride)
		self.stride = stride

	def img2col(self, x):
		col_matrix = []
		channel, height, width = self.in_shape
		for i in range(0, height-self.k_size+1, self.stride):
			for j in range(0, width-self.k_size+1, self.stride):
				#convert kernel size squre into row
				col_matrix.append(x[:, i:i+self.k_size, j:j+self.k_size].reshape([-1]))
		return np.array(col_matrix)

	def forward(self, x):
		act = []
		self.input = []
		for i in range(x.shape[0]):
			self.input.append(self.img2col(x[i]))
			out = self.input[i].dot(self.w)+self.b
			act.append(out.T.reshape(self.out_shape))
		return self.act_func(np.array(act))

	def col2img(self, grad_colin):
		k_size = self.k_size
		img = np.zeros(self.in_shape)
		for row in range(grad_colin.shape[0]):
			i = row//self.out_shape[2]*self.stride
			j = row%self.out_shape[2]*self.stride
			img[:, i:i+k_size, j:j+k_size] += grad_colin[row].reshape([self.in_shape[0], k_size, k_size])
		return img

	def gradient(self, grad_act, act):
		batch_size = grad_act.shape[0]
		b_vec = np.ones((1, self.out_shape[1] * self.out_shape[2]))
		grad_out = self.dact_func(grad_act, act).reshape([batch_size, self.out_shape[0], -1])
		self.grad_w = np.zeros(self.w.shape)
		self.grad_b = np.zeros(self.b.shape)
		grad_in = []
		for i in range(batch_size):
			grad_out_i = grad_out[i].T
			self.grad_w += self.input[i].T.dot(grad_out_i)
			self.grad_b += b_vec.dot(grad_out_i)
			grad_in.append(self.col2img(grad_out_i.dot(self.w.T)))
		self.grad_w /= batch_size
		self.grad_b /= batch_size
		self.input = None
		return np.array(grad_in)

class MaxPooling(Layer):
	def __init__(self, in_shape, k_size, stride=None):
		super(MaxPooling, self).__init__(has_param=False)
		self.in_shape = in_shape
		channel, height, width = in_shape
		self.k_size = k_size
		self.stride = k_size if stride is None else stride 
		self.out_shape = [channel, height//self.stride, width//self.stride]

	def forward(self, x):
		batch_size = x.shape[0]
		channel, height, width = self.in_shape
		self.mask = np.zeros((batch_size, channel, height, width))
		out = np.zeros((batch_size, channel, self.out_shape[1], self.out_shape[2]))
		for b_idx in range(batch_size):
			for c_idx in range(channel):
				for i in range(0, height-self.k_size+1, self.stride):
					for j in range(0, width-self.k_size+1, self.stride):
						out[b_idx, c_idx, i//self.stride, j//self.stride] = \
							np.max(x[b_idx, c_idx, i:i+self.k_size, j:j+self.k_size])
						max_idx = np.argmax(x[b_idx, c_idx, i:i+self.k_size, j:j+self.k_size])
						self.mask[b_idx, c_idx, i + max_idx//self.k_size, j + max_idx%self.k_size] = 1
		return out

	def gradient(self, grad_out):
		grad_out = np.repeat(grad_out, self.k_size, axis=2)
		grad_out = np.repeat(grad_out, self.k_size, axis=3)
		return np.multiply(self.mask, grad_out)


class Softmax(Layer):
	def __init__(self, w_size):
		super(Softmax, self).__init__(has_param=False)

	def forward(self, x):
		return self.predict(x)

	def loss(self, out, y):
		eps = 1e-8
		return -(np.multiply(y, np.log(out+eps))).mean()

	def predict(self, x):
		eps = 1e-8
		out = np.exp(x - np.max(x, axis=1).reshape([-1, 1]))
		return out / (np.sum(out, axis=1).reshape([-1, 1]) + eps)

	def gradient(self, out, y):
		return out - y

class FullyConnect(Layer):
	def __init__(self, in_shape, out_dim, act_type):
		super(FullyConnect, self).__init__(has_param=True)
		self.act_func = self.act_funcs[act_type]
		self.dact_func = self.dact_funcs[act_type]
		self.in_shape = in_shape
		self.w = np.random.randn(in_shape[0]*in_shape[1]*in_shape[2], out_dim)
		self.b = np.random.randn(1, out_dim)
		self.mom_w = np.zeros_like(self.w)
		self.cache_w = np.zeros_like(self.w)
		self.mom_b = np.zeros_like(self.b)
		self.cache_b = np.zeros_like(self.b)

	def forward(self, x):
		self.input = x.reshape([x.shape[0],-1])
		return self.act_func(self.input.dot(self.w)+self.b)

	def gradient(self, grad_act, act):
		grad_act=grad_act.reshape([grad_act.shape[0], grad_act.shape[1]])
		batch_size = grad_act.shape[0]
		grad_out = self.dact_func(grad_act, act)
		self.grad_w = self.input.T.dot(grad_out)
		self.grad_b = np.ones((1, batch_size)).dot(grad_out)
		self.grad_w /= batch_size
		self.grad_b /= batch_size
		self.input = None
		return grad_out.dot(self.w.T).reshape([-1, self.in_shape[0], self.in_shape[1], self.in_shape[2]])