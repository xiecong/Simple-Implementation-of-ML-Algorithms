import numpy as np
import matplotlib.pyplot as plt

def gen_linear_data(train_num):
	x = 2 * np.random.random((train_num, 2)) - 1
	y = np.array([[1] if(xi.sum() > 0) else [-1] for xi in x])
	return x, y

def gen_circle_data(train_num):
	x = 2 * np.random.random((train_num, 2)) - 1
	y = np.array([[1] if(np.square(xi).sum() > 0.6) else [-1] for xi in x])
	return x, y

def gen_xor_data(train_num):
	x = 2 * np.random.random((train_num, 2)) - 1
	y = np.array([[1] if(xi[0]*xi[1]>0) else [-1] for xi in x])
	return x, y

def gen_spiral_data(train_num):
	r = np.arange(train_num) / train_num
	c = np.arange(train_num)%2
	t = 1.75 * r * 2 * np.pi + c * np.pi;
	y = c.reshape(train_num,1)*2-1
	x = np.c_[r * np.sin(t),r * np.cos(t)]
	return x, y

# visualize decision boundary change
def boundary_vis(model, x, y, epoch=1, subplot=[1, 1, 1]):
	clabel = ['red' if yi[0] < 0 else 'blue' for yi in y]
	#loss = np.square(model.predict(x) - y).sum() / 2
	plt.subplot(subplot[0], subplot[1], subplot[2])
	xx, yy = np.meshgrid(np.linspace(-1, 1, 50), np.linspace(-1, 1, 50))
	zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
	#plt.title("epoch {}, loss={}".format(epoch, loss))
	plt.contourf(xx, yy, zz, levels=np.linspace(zz.min(), zz.max(), 40), cmap=plt.cm.RdBu)
	plt.contour(xx, yy, zz, levels=[0], colors='darkred')
	plt.scatter(x[:, 0], x[:, 1], c=clabel, s=10, edgecolors='k')
	if subplot[2] == subplot[0] * subplot[1]: 
		plt.show()
