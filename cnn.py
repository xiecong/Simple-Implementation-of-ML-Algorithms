import numpy as np
from sklearn.datasets import fetch_openml
from cnn_layer_advanced import *
#This implements Lenet-5, test on MNIST dataset

class CNN(object):
	def __init__(self, x_shape, label_num):
		self.batch_size = 100
		self.conv1 = Conv(in_shape=x_shape[1:4], k_num=6, k_size=5, act_type="Sigmoid")
		self.pool1 = MaxPooling(in_shape=self.conv1.out_shape, k_size=2)
		self.conv2 = Conv(in_shape=self.pool1.out_shape, k_num=16, k_size=3, act_type="Sigmoid")
		self.pool2 = MaxPooling(in_shape=self.conv2.out_shape, k_size=2)
		self.fc1 = FullyConnect(self.pool2.out_shape, 120, act_type="Sigmoid")
		self.fc2 = FullyConnect([120], 84, act_type="Sigmoid")
		self.fc3 = FullyConnect([84], label_num, act_type="Linear")
		self.softmax = Softmax(label_num)

	def fit(self, train_x, train_y, test_x, test_y):
		for epoch in range(5):
			#mini batch
			permut=np.random.permutation(train_x.shape[0]//self.batch_size*self.batch_size).reshape([-1,self.batch_size])
			for b_idx in range(permut.shape[0]):
				x0 = train_x[permut[b_idx,:]]
				y = train_y[permut[b_idx,:]]
				out_c1 = self.conv1.forward(x0)
				out_p1 = self.pool1.forward(out_c1)
				out_c2 = self.conv2.forward(out_p1)
				out_p2 = self.pool2.forward(out_c2)
				out_fc1 = self.fc1.forward(out_p2)
				out_fc2 = self.fc2.forward(out_fc1)
				out_fc3 = self.fc3.forward(out_fc2)
				out_sf = self.softmax.forward(out_fc3)

				print("epoch {} batch {} loss: {}".format(epoch, b_idx, self.softmax.loss(out_sf, y)))

				grad_sf = self.softmax.gradient(out_sf, y)
				grad_fc3 = self.fc3.gradient(grad_sf, out_fc3)
				grad_fc2 = self.fc2.gradient(grad_fc3, out_fc2)
				grad_fc1 = self.fc1.gradient(grad_fc2, out_fc1)
				grad_p2 = self.pool2.gradient(grad_fc1)
				grad_c2 = self.conv2.gradient(grad_p2, out_c2)
				grad_p1 = self.pool1.gradient(grad_c2)
				grad_c1 = self.conv1.gradient(grad_p1, out_c1)

				self.conv1.backward("Adam")
				self.conv2.backward("Adam")
				self.fc3.backward("Adam")
				self.fc2.backward("Adam")
				self.fc1.backward("Adam")
			self.test(test_x, test_y)

	def test(self, test_x, test_y):
		loss = 0
		acc = 0
		size = test_y.shape[0]
		for i in range(0, len(test_x), self.batch_size):
			x = test_x[i:i+self.batch_size]
			y = test_y[i:i+self.batch_size]
			preds = self.predict(x)
			loss += self.softmax.loss(preds, y) * x.shape[0]
			acc += sum(np.argmax(y[i])==np.argmax(o) for i, o in enumerate(preds))
		print("loss: {} accuracy: {}".format(loss/size, acc/size))

	def predict(self, x):
		out_c1 = self.conv1.forward(x)
		out_p1 = self.pool1.forward(out_c1)
		out_c2 = self.conv2.forward(out_p1)
		out_p2 = self.pool2.forward(out_c2)
		out_fc1 = self.fc1.forward(out_p2)
		out_fc2 = self.fc2.forward(out_fc1)
		out_fc3 = self.fc3.forward(out_fc2)
		return self.softmax.forward(out_fc3)

	def gradient_check(self):
		conva = Conv(in_shape=[16,32,28], k_num=12, k_size=3, act_type="Tanh")
		convb = Conv(in_shape=[16,32,28], k_num=12, k_size=3, act_type="Tanh")
		convb.w = conva.w.copy()
		convb.b = conva.b.copy()
		eps = 1e-4
		x = np.random.randn(10,16,32,28)*10
		x_a = x.copy()
		x_b = x.copy()
		idxes = [2,10,14,15]
		x_a[idxes[0],idxes[1],idxes[2],idxes[3]]+=eps
		x_b[idxes[0],idxes[1],idxes[2],idxes[3]]-=eps
		out = conva.forward(x)
		gradient = conva.gradient(np.ones(out.shape), out)
		# the output should be in the order of eps*eps 
		print((conva.forward(x_a) - convb.forward(x_b)).sum()/eps/2-gradient[idxes[0],idxes[1],idxes[2],idxes[3]])

def main():
	x, label = fetch_openml('mnist_784', return_X_y=True, data_home="data")
	x = x.reshape(x.shape[0],1,28,28)
	y = np.zeros((x.shape[0],10))
	y[np.arange(x.shape[0]), label.astype(np.int_)] = 1

	train_num = 60000
	cnn = CNN((train_num,1,28,28), 10)
	cnn.fit(x[:train_num], y[:train_num], x[train_num:], y[train_num:])
	
if __name__ == "__main__":
	main()