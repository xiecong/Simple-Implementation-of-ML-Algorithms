import numpy as np
from sklearn.datasets import fetch_openml
from nn_layers import FullyConnect, Activation, Softmax, BatchNormalization, Conv, TrasposedConv
import matplotlib.pyplot as plt
# TO DO dcgan


def noise(n_x, n_d):
	return np.random.randn(n_x, n_d)	

def bce_loss(pred, y):
	eps = 1e-20
	return -((1 - y) * np.log(1 - pred + eps) + y * np.log(pred + eps)).mean()

def bce_grad(pred, y):
	eps = 1e-20
	return (- y / (pred + eps) + (1 - y) / (1 - pred + eps)) / pred.shape[0]


class NN(object):
	def __init__(self, layers):
		self.layers = layers

	def predict(self, x):
		out = x
		for layer in self.layers:
			out = layer.predict_forward(out) if isinstance(layer, BatchNormalization) else layer.forward(out)
		return out

	def forward(self, x):
		out = x
		for layer in self.layers:
			out = layer.forward(out)
		return out

	def gradient(self, grad_loss):
		grad = grad_loss
		for layer in self.layers[::-1]:
			grad = layer.gradient(grad)
		return grad

	def backward(self):
		for layer in self.layers:
			layer.backward()

class GAN(object):
	def __init__(self):
		self.n_epochs, self.batch_size = 5, 32
		self.gen_input = 100
		dc_gan()
		
	def vanilla_gan():
		gen_lr, dis_lr = 2e-3, 5e-4
		self.generator = NN([
			FullyConnect([self.gen_input], [256], lr=gen_lr),
			BatchNormalization([256], lr=gen_lr),
			Activation(act_type='ReLU'),
			FullyConnect([256], [512], lr=gen_lr),
			BatchNormalization([512], lr=gen_lr),
			Activation(act_type='ReLU'),
			FullyConnect([512], [1024], lr=gen_lr),
			BatchNormalization([1024], lr=gen_lr),
			Activation(act_type='ReLU'),
			FullyConnect([1024], [784], lr=gen_lr),
			BatchNormalization([784], lr=gen_lr),
			Activation(act_type='Tanh')
		])
		self.discriminator = NN([
			FullyConnect([784], [1024], lr=dis_lr),
			Activation(act_type='ReLU'),
			FullyConnect([1024], [512], lr=dis_lr),
			Activation(act_type='ReLU'),
			FullyConnect([512], [256], lr=dis_lr),
			Activation(act_type='ReLU'),
			FullyConnect([256], [1], lr=dis_lr),
			Activation(act_type='Sigmoid')
		])

	def dc_gan():
		gen_lr, dis_lr = 2e-3, 5e-4
		pass

	def fit(self, x):
		y_dis = np.zeros((2 * self.batch_size, 1))
		y_dis[:self.batch_size, 0] = 1
		y_gen = np.ones((2 * self.batch_size, 1))
		generated_img = []

		for epoch in range(self.n_epochs):
			permut=np.random.permutation(x.shape[0]//self.batch_size*self.batch_size).reshape([-1,self.batch_size])
			for b_idx in range(permut.shape[0]):
				x_dis_train = np.concatenate([
					x[permut[b_idx,:]], self.generator.forward(noise(self.batch_size, self.gen_input))
				], axis=0)
				pred_dis = self.discriminator.forward(x_dis_train)
				self.discriminator.gradient(bce_grad(pred_dis, y_dis))
				self.discriminator.backward()

				x_gen_train = self.generator.forward(noise(2*self.batch_size, self.gen_input))
				pred_gen = self.discriminator.forward(x_gen_train)

				grad = self.discriminator.gradient(bce_grad(pred_gen, y_gen))
				self.generator.gradient(grad)
				self.generator.backward()
				print(f'Epoch {epoch} batch {b_idx} discriminator:', bce_loss(pred_dis, y_dis), 'generator:', bce_loss(pred_gen, y_gen))

			generated_img.append(self.generator.predict(noise(10, self.gen_input)))
		return generated_img


def main():
	x, _ = fetch_openml('mnist_784', return_X_y=True, data_home='data')
	x = 2 * (x / x.max()) - 1
	gan = GAN()
	images = gan.fit(x)
	for i, img in enumerate(np.array(images).reshape(-1, 784)):
		plt.subplot(len(images), 10, i + 1)
		plt.imshow(img.reshape(28, 28), cmap='gray', vmin=-1, vmax=1)
	plt.show()

if __name__ == "__main__":
	main()