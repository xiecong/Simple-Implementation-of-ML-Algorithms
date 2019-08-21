import numpy as np
from sklearn.datasets import fetch_openml
from cnn_layer_advanced import Conv, MaxPooling, FullyConnect, Activation, Softmax, BatchNormalization
# This implements Lenet-4, test on MNIST dataset
# gradient check for all layers for input x, w, b

class CNN(object):
	def __init__(self, x_shape, label_num):
		self.batch_size = 32
		# Conv > Normalization > Activation > Dropout > Pooling
		self.conv1 = Conv(in_shape=x_shape, k_num=6, k_size=5)
		self.bn1 = BatchNormalization(in_shape=self.conv1.out_shape)
		self.relu1 = Activation(in_shape=self.conv1.out_shape, act_type="ReLU")
		self.pool1 = MaxPooling(in_shape=self.conv1.out_shape, k_size=2)
		self.conv2 = Conv(in_shape=self.pool1.out_shape, k_num=16, k_size=3)
		self.bn2 = BatchNormalization(in_shape=self.conv2.out_shape)
		self.relu2 = Activation(in_shape=self.conv2.out_shape, act_type="ReLU")
		self.pool2 = MaxPooling(in_shape=self.conv2.out_shape, k_size=2)
		self.fc1 = FullyConnect(self.pool2.out_shape, 120)
		self.bn3 = BatchNormalization(in_shape=[120])
		self.relu3 = Activation(in_shape=[120], act_type="ReLU")
		self.fc2 = FullyConnect([120], label_num)
		self.softmax = Softmax(label_num)

	def fit(self, train_x, labels):
		n_data = train_x.shape[0]
		train_y = np.zeros((n_data, 10))
		train_y[np.arange(n_data), labels] = 1
		for epoch in range(3):
			#mini batch
			permut=np.random.permutation(n_data//self.batch_size*self.batch_size).reshape([-1,self.batch_size])
			for b_idx in range(permut.shape[0]):
				x0 = train_x[permut[b_idx,:]]
				y = train_y[permut[b_idx,:]]
				out_c1 = self.conv1.forward(x0)
				out_bn1 = self.bn1.forward(out_c1)
				out_r1 = self.relu1.forward(out_bn1)
				out_p1 = self.pool1.forward(out_r1)
				out_c2 = self.conv2.forward(out_p1)
				out_bn2 = self.bn2.forward(out_c2)
				out_r2 = self.relu2.forward(out_bn2)
				out_p2 = self.pool2.forward(out_r2)
				out_fc1 = self.fc1.forward(out_p2)
				out_bn3 = self.bn3.forward(out_fc1)
				out_r3 = self.relu3.forward(out_bn3)
				out_fc2 = self.fc2.forward(out_r3)
				out_sf = self.softmax.forward(out_fc2)
				if b_idx%100==0:
					print("epoch {} batch {} loss: {}".format(epoch, b_idx, self.softmax.loss(out_sf, y)))

				grad_sf = self.softmax.gradient(out_sf, y)
				grad_fc2 = self.fc2.gradient(grad_sf)
				grad_r3 = self.relu3.gradient(grad_fc2, out_r3)
				grad_bn3 = self.bn3.gradient(grad_r3)
				grad_fc1 = self.fc1.gradient(grad_bn3)
				grad_p2 = self.pool2.gradient(grad_fc1)
				grad_r2 = self.relu2.gradient(grad_p2, out_r2)
				grad_bn2 = self.bn2.gradient(grad_r2)
				grad_c2 = self.conv2.gradient(grad_bn2)
				grad_p1 = self.pool1.gradient(grad_c2)
				grad_r1 = self.relu1.gradient(grad_p1, out_r1)
				grad_bn1 = self.bn1.gradient(grad_r1)
				grad_c1 = self.conv1.gradient(grad_bn1)

				self.conv1.backward("Adam")
				self.bn1.backward("Adam")
				self.conv2.backward("Adam")
				self.bn2.backward("Adam")
				self.fc2.backward("Adam")
				self.bn3.backward("Adam")
				self.fc1.backward("Adam")
			print('acc', self.get_accuracy(train_x, labels))

	def predict(self, x):
		out_c1 = self.conv1.forward(x)
		out_bn1 = self.bn1.forward(out_c1)
		out_r1 = self.relu1.forward(out_bn1)
		out_p1 = self.pool1.forward(out_r1)
		out_c2 = self.conv2.forward(out_p1)
		out_bn2 = self.bn2.forward(out_c2)
		out_r2 = self.relu2.forward(out_bn2)
		out_p2 = self.pool2.forward(out_r2)
		out_fc1 = self.fc1.forward(out_p2)
		out_bn3 = self.bn3.forward(out_fc1)
		out_r3 = self.relu3.forward(out_bn3)
		out_fc2 = self.fc2.forward(out_r3)
		return self.softmax.forward(out_fc2)

	def get_accuracy(self, x, label):
		n_correct = 0
		for i in range(0, x.shape[0], self.batch_size):
			x_batch, label_batch = x[i: i + self.batch_size], label[i: i + self.batch_size]
			n_correct += sum(np.argmax(self.predict(x_batch), axis=1) == label_batch)
		return n_correct / x.shape[0]

def gradient_check(conv=True):
	if conv:
		layera = Conv(in_shape=[16,32,28], k_num=12, k_size=3)
		layerb = Conv(in_shape=[16,32,28], k_num=12, k_size=3)
		act_layer = Activation(in_shape=layera.out_shape, act_type='Tanh')
	else:
		layera = FullyConnect(in_shape=[16,32,28], out_dim=12)
		layerb = FullyConnect(in_shape=[16,32,28], out_dim=12)
		act_layer = Activation(in_shape=[12], act_type='Tanh')
	layerb.w = layera.w.copy()
	layerb.b = layera.b.copy()
	eps = 1e-4
	x = np.random.randn(10,16,32,28) * 10
	for i in range(100):
		idxes = tuple((np.random.uniform(0,1,4) * x.shape).astype(int))
		x_a = x.copy()
		x_b = x.copy()
		x_a[idxes] += eps
		x_b[idxes] -= eps
		out = act_layer.forward(layera.forward(x))
		delta_out = (act_layer.forward(layera.forward(x_a)) - act_layer.forward(layerb.forward(x_b))).sum()
		gradient = layera.gradient(act_layer.gradient(np.ones(out.shape), out))
		# the output should be in the order of eps*eps 
		print(idxes, (delta_out / eps / 2 - gradient[idxes]) / eps / eps)


def main():
	x, y = fetch_openml('mnist_784', return_X_y=True, data_home="data")
	x = x.reshape(-1, 1, 28, 28)

	test_ratio = 0.2
	test_split = np.random.uniform(0, 1, x.shape[0])
	train_x, train_y = x[test_split >= test_ratio] / x.max(), y.astype(np.int_)[test_split >= test_ratio]
	test_x, test_y = x[test_split < test_ratio] / x.max(), y.astype(np.int_)[test_split < test_ratio]

	cnn = CNN(x.shape[1:4], 10)
	cnn.fit(train_x, train_y)
	print('train acc', cnn.get_accuracy(train_x, train_y))
	print('test acc', cnn.get_accuracy(test_x, test_y))

if __name__ == "__main__":
	#gradient_check()
	main()