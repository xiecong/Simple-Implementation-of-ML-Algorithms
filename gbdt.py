import numpy as np
from sklearn.datasets import load_boston
from decision_tree import DecisionTree


def square_loss(y, pred):
	return np.square(pred - y).mean()/2

def gradient(y, pred):
	return pred - y


class GBDT(object):
	def __init__(self):
		self.max_depth = 4
		self.tree_num = 20
		self.forest = []
		self.rhos = []
		self.t0 = 0

	def get_importance(self):
		return sum(tree.get_importance() for tree in self.forest)/self.tree_num

	def _linear_search(self, y, pred, f):
		shrinkage = 1	
		losses = [square_loss(y, pred+i*f/10) for i in range(1,21)]
		return shrinkage * (np.argmin(losses)+1)/10

	def fit(self, x, y):
		self.t0 = y.mean()  # t0, which is a constant
		pred = y.mean()
		for i in range(self.tree_num):
			grad = gradient(y, pred)
			self.forest.append(DecisionTree(metric_type="Variance", depth=self.max_depth, regression=True))
			self.forest[i].fit(x, grad)
			f = np.array([self.forest[i].predict(xi) for xi in x])
			# find best learning rate
			self.rhos.append(self._linear_search(y, pred, f))
			pred -= f * self.rhos[i]
			# for categorical dataset, use cross entropy loss
			print("tree {} constructed, loss {}".format(i, square_loss(y, pred)))

	def predict(self, x):
		return self.t0 - sum(np.array([tree.predict(xi)for xi in x]) * rho for tree, rho in zip(self.forest, self.rhos))


def main():
	data = load_boston()
	x = data.data
	y = data.target

	test_ratio = 0.2
	test_split = np.random.uniform(0, 1, len(x))
	train_x = x[test_split >= test_ratio]
	test_x = x[test_split < test_ratio]
	train_y = y[test_split >= test_ratio]
	test_y = y[test_split < test_ratio]

	gbdt = GBDT()
	gbdt.fit(train_x, train_y)
	print(gbdt.get_importance())
	print(square_loss(test_y, gbdt.predict(test_x)))


if __name__ == "__main__":
	main()