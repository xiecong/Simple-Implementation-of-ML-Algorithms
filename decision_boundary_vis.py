import numpy as np
import matplotlib.pyplot as plt
from mlp_advanced import MLP
from gbdt import GBDT
from xgboost import XGBoost
from random_forest import RandomForest
from adaboost import AdaBoost
from factorization_machines import FactorizationMachines
from svm import SVM
from knn import kNearestNeighbor

def gen_linear(train_num):
	x = 2 * np.random.random((train_num, 2)) - 1
	return x, (x.sum(axis=1) > 0) * 1

def gen_circle(train_num):
	x = 2 * np.random.random((train_num, 2)) - 1
	return x, (np.square(x).sum(axis=1) > 0.6) * 1

def gen_xor(train_num):
	x = 2 * np.random.random((train_num, 2)) - 1
	return x, np.array([(xi[0]*xi[1] > 0) for xi in x]) * 1

def gen_spiral(train_num):
	r = 0.8 * np.arange(train_num) / train_num
	y = np.arange(train_num)%2
	t = 1.75 * r * 2 * np.pi + y * np.pi;
	x = np.c_[r * np.sin(t) + np.random.random(train_num)/10, r * np.cos(t) + np.random.random(train_num)/10]
	return x, y * 1

def gen_moon(train_num):
	y = np.arange(train_num) % 2
	x0 = (y - 0.5) * (.5 - np.cos(np.linspace(0, np.pi, train_num))) + np.random.random(train_num) / 10
	x1 = (y - 0.5) * (.5 - 2 * np.sin(np.linspace(0, np.pi, train_num))) + np.random.random(train_num) / 10
	return np.c_[x0, x1], y

# visualize decision boundary change
def boundary_vis_plots(model, x, y, subplot=[1, 1, 1]):
	plt.subplot(subplot[0], subplot[1], subplot[2])
	xx, yy = np.meshgrid(np.linspace(-1, 1, 50), np.linspace(-1, 1, 50))
	pred = model.predict(np.c_[xx.ravel(), yy.ravel()])
	zz = pred.reshape(xx.shape) if len(pred.shape) == 1 or pred.shape[1] == 1 else pred[:, 1].reshape(xx.shape)
	if subplot[2] <= subplot[1]:
		plt.title(type(model).__name__)
	plt.contourf(xx, yy, zz, levels=np.linspace(zz.min(), zz.max(), 40), cmap=plt.cm.RdBu)
	plt.contour(xx, yy, zz, levels=[0.5], colors='darkred')
	plt.scatter(x[:, 0], x[:, 1], c=np.array(['red', 'blue'])[y], s=10, edgecolors='k')
	if subplot[2] == subplot[0] * subplot[1]: 
		plt.show()


def main():
	data_loaders = [gen_linear, gen_circle, gen_xor, gen_spiral, gen_moon]
	models = [
		(kNearestNeighbor, {'k': 5}),
		(FactorizationMachines, {'learning_rate': 1, 'embedding_dim': 1}),
		(SVM, {}),
		(AdaBoost, {'esti_num': 10}),
		(RandomForest, {'tree_num': 20, 'max_depth': 3}),
		(XGBoost, {'tree_num': 20, 'max_depth': 3}),
		(MLP, {'act_type': 'Tanh', 'opt_type': 'Adam', 'layers': [2, 8, 7, 2], 'epochs': 200, 'learning_rate': 0.5, 'lmbda': 1e-4})
	]
	for i, data_loader in enumerate(data_loaders):
		x, y = data_loader(256)
		for j, model in enumerate(models):
			clf = model[0](**model[1])
			clf.fit(x, y if not j in [2, 3] else 2 * y - 1)
			boundary_vis_plots(clf, x, y, subplot=[len(data_loaders), len(models), len(models) * i + 1 + j])


if __name__ == "__main__":
	main()