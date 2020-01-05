import numpy as np
from sklearn.datasets import load_digits, fetch_openml
from nn_layers import FullyConnect, Activation
import matplotlib.pyplot as plt


class VAE(object):
	def __init__(self, dim_in, dim_hidden, dim_z):
		self.n_epochs, self.batch_size = 10, 32
		self.C = 1  # trade off of reconstruction and KL divergence

		# architecture is hard-coded
		self.encoder_hidden = FullyConnect([dim_in], [dim_hidden], lr=1e-2)
		self.encoder_act = Activation(act_type='ReLU')
		self.encoder_mu = FullyConnect([dim_hidden], [dim_z], lr=1e-2)
		self.encoder_log_sigma = FullyConnect([dim_hidden], [dim_z], lr=1e-2)

		self.decoder_hidden = FullyConnect([dim_z], [dim_hidden], lr=1e-2)
		self.decoder_act_hidden = Activation(act_type='ReLU')
		self.decoder_out = FullyConnect([dim_hidden], [dim_in], lr=1e-2)
		self.decoder_act_out = Activation(act_type='Sigmoid')

	def fit(self, x):
		for epoch in range(self.n_epochs):
			permut=np.random.permutation(
				x.shape[0] // self.batch_size * self.batch_size
			).reshape([-1, self.batch_size])
			for b_idx in range(permut.shape[0]):
				x_batch = x[permut[b_idx,:]]
				mu, log_sigma = self.encoder_forward(x_batch)
				z = self.sampling(mu, log_sigma)
				out = self.decoder_forward(z)

				recon_grad = self.C * (out - x_batch)
				grad_d_act_out = self.decoder_act_out.gradient(recon_grad)
				grad_d_out = self.decoder_out.gradient(grad_d_act_out)
				grad_d_act_hidden = self.decoder_act_hidden.gradient(grad_d_out)
				grad_z = self.decoder_hidden.gradient(grad_d_act_hidden)

				kl_mu_grad = mu
				kl_sigma_grad = np.exp(2 * log_sigma) - 1
				grad_mu = self.encoder_mu.gradient(grad_z + kl_mu_grad)
				grad_log_sigma = self.encoder_log_sigma.gradient(grad_z + kl_sigma_grad)
				grad_e_act = self.encoder_act.gradient(grad_mu + grad_log_sigma)
				grad_e_hidden = self.encoder_hidden.gradient(grad_e_act)

				self.backward()
			print('epoch: {}, log loss: {}, kl loss: {}'.format(
				epoch, self.log_loss(out, x_batch), self.kl_loss(mu, log_sigma)
			))

	def encoder_forward(self, x):
		hidden = self.encoder_hidden.forward(x)
		hidden = self.encoder_act.forward(hidden)
		mu = self.encoder_mu.forward(hidden)
		log_sigma = self.encoder_log_sigma.forward(hidden)
		return mu, log_sigma

	def sampling(self, mu, log_sigma):
		noise = np.random.randn(mu.shape[0], mu.shape[1])
		return mu + noise * np.exp(log_sigma)

	def decoder_forward(self, z):
		hidden = self.decoder_hidden.forward(z)
		hidden = self.decoder_act_hidden.forward(hidden)
		out = self.decoder_out.forward(hidden)
		out = self.decoder_act_out.forward(out)
		return out

	def backward(self):
		self.decoder_act_out.backward()
		self.decoder_out.backward()
		self.decoder_act_hidden.backward()
		self.decoder_hidden.backward()
		self.encoder_mu.backward()
		self.encoder_log_sigma.backward()
		self.encoder_act.backward()
		self.encoder_hidden.backward()

	def log_loss(self, pred, x):
		return 0.5 * self.C  * np.square(pred - x).mean()

	def kl_loss(self, mu, log_sigma):
		return 0.5 * (-2 * log_sigma + np.exp(2 * log_sigma) + np.square(mu) - 1).mean()

def main():
	#data = load_digits()
	#x, y = data.data, data.target
	x, _ = fetch_openml('mnist_784', return_X_y=True, data_home="data")
	vae = VAE(x.shape[1], 64, 2)
	vae.fit(x / x.max())

	n_rows = 11
	for i in range(n_rows):
		for j in range(n_rows):
			plt.subplot(n_rows, n_rows, i * n_rows + j + 1)
			plt.imshow(
				vae.decoder_forward(np.array([[(i - n_rows // 2) / 2, (j - n_rows // 2) / 2]])).reshape(28, 28),
				cmap='gray', vmin=0, vmax=1
			)
	plt.show()


if __name__ == "__main__":
	main()