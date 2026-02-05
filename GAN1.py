"""Small GAN demo.
If TensorFlow is available, uses the original Keras flow; otherwise falls back
to a lightweight PyTorch implementation so `python ./GAN1.py` runs without TF.
"""
import importlib
import importlib.util
import numpy as np
from numpy import hstack, zeros, ones
from numpy.random import rand, randn
from matplotlib import pyplot


def _has_tensorflow() -> bool:
	return importlib.util.find_spec("tensorflow") is not None


if _has_tensorflow():
	from keras.models import Sequential
	from keras.layers import Dense

	def define_discriminator(n_inputs=2):
		model = Sequential()
		model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
		model.add(Dense(1, activation='sigmoid'))
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		return model

	def define_generator(latent_dim, n_outputs=2):
		model = Sequential()
		model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
		model.add(Dense(n_outputs, activation='linear'))
		return model

	def generate_real_samples(n):
		X1 = rand(n) - 0.5
		X2 = X1 * X1
		X1 = X1.reshape(n, 1)
		X2 = X2.reshape(n, 1)
		X = hstack((X1, X2))
		y = ones((n, 1))
		return X, y

	def generate_latent_points(latent_dim, n):
		x_input = randn(latent_dim * n)
		x_input = x_input.reshape(n, latent_dim)
		return x_input

	def generate_fake_samples(generator, latent_dim, n):
		x_input = generate_latent_points(latent_dim, n)
		X = generator.predict(x_input)
		y = zeros((n, 1))
		return X, y

	def summarize_performance(epoch, generator, discriminator, latent_dim, n=100):
		x_real, y_real = generate_real_samples(n)
		_, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
		x_fake, y_fake = generate_fake_samples(generator, latent_dim, n)
		_, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
		print(epoch, acc_real, acc_fake)
		pyplot.scatter(x_real[:, 0], x_real[:, 1], color='red')
		pyplot.scatter(x_fake[:, 0], x_fake[:, 1], color='blue')
		pyplot.show()

	def train(g_model, d_model, gan_model, latent_dim, n_epochs=10000, n_batch=128, n_eval=2000):
		half_batch = int(n_batch / 2)
		for i in range(n_epochs):
			x_real, y_real = generate_real_samples(half_batch)
			x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			d_model.train_on_batch(x_real, y_real)
			d_model.train_on_batch(x_fake, y_fake)
			x_gan = generate_latent_points(latent_dim, n_batch)
			y_gan = ones((n_batch, 1))
			gan_model.train_on_batch(x_gan, y_gan)
			if (i + 1) % n_eval == 0:
				summarize_performance(i, g_model, d_model, latent_dim)

	if __name__ == '__main__':
		latent_dim = 5
		discriminator = define_discriminator()
		generator = define_generator(latent_dim)
		discriminator.trainable = False
		from keras.models import Sequential as _Seq
		gan_model = _Seq()
		gan_model.add(generator)
		gan_model.add(discriminator)
		gan_model.compile(loss='binary_crossentropy', optimizer='adam')
		train(generator, discriminator, gan_model, latent_dim, 20000, 256, 5000)

else:
	import torch
	import torch.nn as nn
	import torch.optim as optim

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	class Discriminator(nn.Module):
		def __init__(self, n_inputs=2):
			super().__init__()
			self.model = nn.Sequential(
				nn.Linear(n_inputs, 25),
				nn.ReLU(inplace=True),
				nn.Linear(25, 1),
				nn.Sigmoid()
			)

		def forward(self, x):
			return self.model(x)

	class Generator(nn.Module):
		def __init__(self, latent_dim, n_outputs=2):
			super().__init__()
			self.model = nn.Sequential(
				nn.Linear(latent_dim, 15),
				nn.ReLU(inplace=True),
				nn.Linear(15, n_outputs)
			)

		def forward(self, z):
			return self.model(z)

	def generate_real_samples(n):
		X1 = np.random.rand(n) - 0.5
		X2 = X1 * X1
		X = np.hstack((X1.reshape(n,1), X2.reshape(n,1))).astype(np.float32)
		y = np.ones((n,1), dtype=np.float32)
		return X, y

	def generate_latent_points(latent_dim, n):
		x_input = np.random.randn(latent_dim * n)
		return x_input.reshape(n, latent_dim).astype(np.float32)

	def generate_fake_samples(generator, latent_dim, n):
		x_input = generate_latent_points(latent_dim, n)
		with torch.no_grad():
			z = torch.from_numpy(x_input).to(device)
			X = generator(z).cpu().numpy()
		y = np.zeros((n,1), dtype=np.float32)
		return X, y

	def summarize_performance(epoch, generator, discriminator, latent_dim, n=100):
		x_real, y_real = generate_real_samples(n)
		xr = torch.from_numpy(x_real).to(device)
		yr = torch.from_numpy(y_real).to(device)
		with torch.no_grad():
			acc_real = ((discriminator(xr) > 0.5).float() == yr).float().mean().item()
		x_fake, y_fake = generate_fake_samples(generator, latent_dim, n)
		xf = torch.from_numpy(x_fake).to(device)
		yf = torch.from_numpy(y_fake).to(device)
		with torch.no_grad():
			acc_fake = ((discriminator(xf) > 0.5).float() == yf).float().mean().item()
		print(epoch, acc_real, acc_fake)
		pyplot.scatter(x_real[:, 0], x_real[:, 1], color='red')
		pyplot.scatter(x_fake[:, 0], x_fake[:, 1], color='blue')
		pyplot.show()

	def train(g_model, d_model, latent_dim, n_epochs=10000, n_batch=128, n_eval=2000):
		criterion = nn.BCELoss()
		opt_d = optim.Adam(d_model.parameters(), lr=0.001)
		opt_g = optim.Adam(g_model.parameters(), lr=0.001)
		half_batch = n_batch // 2
		for i in range(n_epochs):
			x_real, y_real = generate_real_samples(half_batch)
			x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)

			xr = torch.from_numpy(x_real).to(device)
			yr = torch.from_numpy(y_real).to(device)
			xf = torch.from_numpy(x_fake).to(device)
			yf = torch.from_numpy(y_fake).to(device)

			d_model.train()
			opt_d.zero_grad()
			pred_real = d_model(xr)
			loss_real = criterion(pred_real, yr)
			pred_fake = d_model(xf)
			loss_fake = criterion(pred_fake, yf)
			loss_d = (loss_real + loss_fake) * 0.5
			loss_d.backward()
			opt_d.step()

			g_model.train()
			opt_g.zero_grad()
			x_gan = generate_latent_points(latent_dim, n_batch)
			z = torch.from_numpy(x_gan).to(device)
			gen_output = g_model(z)
			valid = torch.ones((n_batch,1), dtype=torch.float32, device=device)
			pred = d_model(gen_output)
			loss_g = criterion(pred, valid)
			loss_g.backward()
			opt_g.step()

			if (i + 1) % n_eval == 0:
				summarize_performance(i, g_model, d_model, latent_dim)

	if __name__ == '__main__':
		latent_dim = 5
		D = Discriminator().to(device)
		G = Generator(latent_dim).to(device)
		train(G, D, latent_dim, n_epochs=20000, n_batch=256, n_eval=5000)