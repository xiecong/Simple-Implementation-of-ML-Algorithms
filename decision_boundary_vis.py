import numpy as np
import matplotlib.pyplot as plt
from mlp_advanced import MLP
from gbdt import GBDT
from xgboost import XGBoost
from random_forest import RandomForest
from adaboost import AdaBoost
from factorization_machines import FactorizationMachines
from svm import SVM

def gen_linear(train_num):
	x = 2 * np.random.random((train_num, 2)) - 1
	return x, 2*(x.sum(axis=1) > 0) -1

def gen_circle(train_num):
	x = 2 * np.random.random((train_num, 2)) - 1
	return x, 2*(np.square(x).sum(axis=1) > 0.6)-1

def gen_xor(train_num):
	x = 2 * np.random.random((train_num, 2)) - 1
	return x, 2*np.array([(xi[0]*xi[1] > 0) for xi in x])-1

def gen_spiral(train_num):
	r = np.arange(train_num) / train_num
	y = np.arange(train_num)%2
	t = 1.75 * r * 2 * np.pi + y * np.pi;
	x = np.c_[r * np.sin(t),r * np.cos(t)]
	return x, 2*y-1

# visualize decision boundary change
def boundary_vis_plots(model, x, y, subplot=[1, 1, 1]):
	clabel = ['red' if yi < 0 else 'blue' for yi in y]
	plt.subplot(subplot[0], subplot[1], subplot[2])
	xx, yy = np.meshgrid(np.linspace(-1, 1, 50), np.linspace(-1, 1, 50))
	zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape) - 0.5
	plt.contourf(xx, yy, zz, levels=np.linspace(zz.min(), zz.max(), 40), cmap=plt.cm.RdBu)
	plt.contour(xx, yy, zz, levels=[0], colors='darkred')
	plt.scatter(x[:, 0], x[:, 1], c=clabel, s=10, edgecolors='k')
	if subplot[2] == subplot[0] * subplot[1]: 
		plt.show()

def main():
	x, y = gen_spiral(1000)

	test_ratio = 0.2
	test_split = np.random.uniform(0, 1, len(x))
	train_x = x[test_split >= test_ratio]
	test_x = x[test_split < test_ratio]
	train_y = y[test_split >= test_ratio]
	test_y = y[test_split < test_ratio]

	svm = SVM()
	svm.fit(train_x, train_y)
	boundary_vis_plots(svm, test_x, test_y, subplot=[1, 6, 1])

	mlp = MLP('Tanh', 'Adam', layers=[2, 8, 7, 1], epochs=200, regression=True, learning_rate=0.5, lmbda=1e-4)
	mlp.fit(train_x, train_y.reshape(-1, 1))
	boundary_vis_plots(mlp, test_x, test_y, subplot=[1, 6, 2])

	xgboost = XGBoost(tree_num=20, max_depth=3)
	xgboost.fit(train_x, train_y)
	boundary_vis_plots(xgboost, test_x, test_y, subplot=[1, 6, 3])

	rf = RandomForest(tree_num=20, max_depth=3)
	rf.fit(train_x, train_y)
	boundary_vis_plots(rf, test_x, test_y, subplot=[1, 6, 4])

	adaboost = AdaBoost(esti_num=50)
	adaboost.fit(train_x, train_y)
	boundary_vis_plots(adaboost, test_x, test_y, subplot=[1, 6, 5])

	fm = FactorizationMachines(regression=True, learning_rate=1, embedding_dim=1)
	fm.fit(train_x, train_y)
	boundary_vis_plots(fm, test_x, test_y, subplot=[1, 6, 6])


if __name__ == "__main__":
	main()