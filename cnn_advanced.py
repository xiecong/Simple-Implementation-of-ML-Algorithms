import numpy as np
from sklearn.datasets import fetch_openml

# vectorize all the operation
# dropout
# batch normalization

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

def img2col(x, k_size, stride):
	col_matrix = []
	batch, channel, height, width = x.shape
	for i in range(0, height-k_size+1, stride):
		for j in range(0, width-k_size+1, stride):
			#convert kernel size squre into row
			col_matrix.append(x[:, :, i:i+k_size, j:j+k_size].reshape([batch,-1]))
	return np.array(col_matrix).reshape(-1, channel*k_size*k_size)
	
def col2img(col, in_shape, out_shape, k_size, stride):
	in_c, in_h, in_w = in_shape
	out_c, out_h, out_w = out_shape
	batch_size = col.shape[0]//out_h//out_w
	img = np.zeros((batch_size, in_c, in_h, in_w))
	for row in range(col.shape[0]):
		b_idx = row%batch_size
		pos = row//batch_size
		i = pos//out_w*stride
		j = pos%out_w*stride
		img[b_idx, :, i:i+k_size, j:j+k_size] += col[row].reshape([in_c, k_size, k_size])
	return img

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

	def bp(self, grad_act):
		pass

	def update(self, grad_type):
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

	def img2col2(self, x):
		col_matrix = []
		channel, height, width = self.in_shape
		for i in range(0, height-self.k_size+1, self.stride):
			for j in range(0, width-self.k_size+1, self.stride):
				#convert kernel size squre into row
				col_matrix.append(x[:, i:i+self.k_size, j:j+self.k_size].reshape([-1]))
		return np.array(col_matrix)

	def forward2(self, x):
		act = []
		self.input = []
		for i in range(x.shape[0]):
			self.input.append(self.img2col2(x[i]))
			out = self.input[i].dot(self.w)+self.b
			act.append(out.T.reshape(self.out_shape))
		return self.act_func(np.array(act))

	def forward(self, x):
		self.input = img2col(x, self.k_size, self.stride)
		out = self.act_func(self.input.dot(self.w)+self.b)
		out = out.reshape(self.out_shape[1], self.out_shape[2], x.shape[0], self.out_shape[0])
		return out.transpose(2, 3, 0, 1)

	def col2img2(self, grad_colin):
		k_size = self.k_size
		img = np.zeros(self.in_shape)
		for row in range(grad_colin.shape[0]):
			i = row//self.out_shape[2]*self.stride
			j = row%self.out_shape[2]*self.stride
			img[:, i:i+k_size, j:j+k_size] += grad_colin[row].reshape([self.in_shape[0], k_size, k_size])
		return img

	def bp2(self, grad_act, act):
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
			grad_in.append(self.col2img2(grad_out_i.dot(self.w.T)))
		self.grad_w /= batch_size
		self.grad_b /= batch_size
		self.input = None
		return np.array(grad_in)

	def bp(self, grad_act, act):
		batch_size = grad_act.shape[0]
		grad_out = self.dact_func(grad_act, act).transpose(2,3,0,1).reshape([-1, self.out_shape[0]])
		self.grad_w = self.input.T.dot(grad_out) / batch_size
		self.grad_b = np.ones((1, grad_out.shape[0])).dot(grad_out) / batch_size
		self.input = None
		return np.array(col2img(grad_out.dot(self.w.T), self.in_shape, self.out_shape, self.k_size, self.stride))

class MaxPooling(Layer):
	def __init__(self, in_shape, k_size, stride=None):
		super(MaxPooling, self).__init__(has_param=False)
		self.in_shape = in_shape
		channel, height, width = in_shape
		self.k_size = k_size
		self.stride = k_size if stride is None else stride 
		self.out_shape = [channel, height//self.stride, width//self.stride]

	def bp(self, grad_out):
		grad_out = np.repeat(grad_out, self.k_size, axis=2)
		grad_out = np.repeat(grad_out, self.k_size, axis=3)
		return np.multiply(self.mask, grad_out)

	def forward(self, x):
		col = img2col(x.reshape(-1,1,self.in_shape[1],self.in_shape[2]), k_size=self.k_size, stride=self.stride)
		max_idx = np.argmax(col, axis=1)
		col_mask = np.zeros(col.shape)
		col_mask[range(col.shape[0]),max_idx] = 1
		col_mask = col_mask.reshape(self.out_shape[1]* self.out_shape[2]* x.shape[0], self.in_shape[0]*self.k_size*self.k_size)
		self.mask = col2img(col_mask, self.in_shape, self.out_shape, self.k_size, self.stride)
		out = col[range(col.shape[0]),max_idx].reshape(self.out_shape[1],self.out_shape[2],x.shape[0], self.in_shape[0])
		return out.transpose(2, 3, 0, 1)

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

	def bp(self, out, y):
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

	def bp(self, grad_act, act):
		grad_act=grad_act.reshape([grad_act.shape[0], grad_act.shape[1]])
		batch_size = grad_act.shape[0]
		grad_out = self.dact_func(grad_act, act)
		self.grad_w = self.input.T.dot(grad_out)
		self.grad_b = np.ones((1, batch_size)).dot(grad_out)
		self.grad_w /= batch_size
		self.grad_b /= batch_size
		self.input = None
		return grad_out.dot(self.w.T).reshape([-1, self.in_shape[0], self.in_shape[1], self.in_shape[2]])

class CNN(object):
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.batch_size = 64
		self.conv1 = Conv(in_shape=x.shape[1:4], k_num=12, k_size=5, act_type="ReLU")
		self.pool1 = MaxPooling(in_shape=self.conv1.out_shape, k_size=2)
		self.conv2 = Conv(in_shape=self.pool1.out_shape, k_num=24, k_size=3, act_type="ReLU")
		self.pool2 = MaxPooling(in_shape=self.conv2.out_shape, k_size=2)
		self.fc = FullyConnect(self.pool2.out_shape, y.shape[1], act_type="Linear")
		self.softmax = Softmax(y.shape[1])

	def fit(self):
		for epoch in range(20):
			#mini batch
			permut=np.random.permutation(self.x.shape[0]//self.batch_size*self.batch_size).reshape([-1,self.batch_size])
			for b_idx in range(permut.shape[0]):
				x0 = self.x[permut[b_idx,:]]
				y = self.y[permut[b_idx,:]]
				out_c1 = self.conv1.forward(x0)
				out_p1 = self.pool1.forward(out_c1)
				out_c2 = self.conv2.forward(out_p1)
				out_p2 = self.pool2.forward(out_c2)
				out_fc = self.fc.forward(out_p2)
				out_sf = self.softmax.forward(out_fc)

				print("epoch {} batch {} loss: {}".format(epoch, b_idx, self.softmax.loss(out_sf, y)))

				grad_sf = self.softmax.bp(out_sf, y)
				grad_fc = self.fc.bp(grad_sf, out_fc)
				grad_p2 = self.pool2.bp(grad_fc)
				grad_c2 = self.conv2.bp(grad_p2, out_c2)
				grad_p1 = self.pool1.bp(grad_c2)
				grad_c1 = self.conv1.bp(grad_p1, out_c1)

				self.conv1.update("Adam")
				self.conv2.update("Adam")
				self.fc.update("Adam")
			preds = self.predict(self.x)
			print("epoch {} {}".format(epoch, self.softmax.loss(preds,self.y)))
			print("accuracy: {}".format(sum(np.argmax(self.y[i])==np.argmax(o) for i, o in enumerate(preds))/self.y.shape[0]))

	def predict(self, x):
		out_c1 = self.conv1.forward(x)
		out_p1 = self.pool1.forward(out_c1)
		out_c2 = self.conv2.forward(out_p1)
		out_p2 = self.pool2.forward(out_c2)
		out_fc = self.fc.forward(out_p2)
		return self.softmax.forward(out_fc)

def loadMNIST(prefix, folder):
	intType = np.dtype('int32').newbyteorder('>')
	nMetaDataBytes = 4 * intType.itemsize
	data = np.fromfile(folder + "/" + prefix + '-images.idx3-ubyte', dtype = 'ubyte')
	magicBytes, nImages, width, height = np.frombuffer(data[:nMetaDataBytes].tobytes(), intType)
	data = data[nMetaDataBytes:].astype(dtype = 'float32').reshape([nImages, 1, width, height])
	labels = np.fromfile(folder + "/" + prefix + '-labels.idx1-ubyte', dtype='ubyte')[2 * intType.itemsize:]
	return data, labels

def main():
	x, y = fetch_openml('mnist_784', version=1, cache=True, return_X_y=True, data_home="data")
	print(x.shape, y.shape)
	train_x, train_y = loadMNIST("t10k", "data/mnist")
	train_num = train_x.shape[0]
	train_label = np.zeros((train_num, 10))
	print(train_x.shape)
	train_label[np.arange(train_num), train_y] = 1
	cnn = CNN(train_x, train_label)
	cnn.fit()
	
if __name__ == "__main__":
	main()