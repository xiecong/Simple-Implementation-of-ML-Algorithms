import numpy as np
from sklearn.datasets import load_boston
from decision_tree import DecisionTree


def square_loss(y, pred):
	return np.square(pred-y).mean()/2


class XGBoostRegressionTree(DecisionTree):
	def __init__(self, max_depth):
		self.lambd = 0.01
		self.gamma = 0.1
		super(XGBoostRegressionTree, self).__init__(metric_type="Gini impurity", depth=max_depth, regression=True)
		self.metric = self.score

	def gen_leaf(self, data):
		return {'label': -data[:,-1].sum()/(data.shape[0]+self.lambd)}

	def score(self, data):
		return np.square(data[:,-1].sum())/(data.shape[0]+self.lambd)

	def split_gain(self, p_score, l_child, r_child):
		return self.metric(l_child) + self.metric(r_child) - p_score - self.gamma

# importance for each feature
class XGBoost(object):
	def __init__(self):
		self.max_depth = 4
		self.tree_num = 10
		self.forest = []
		self.shrinkage = 1

	def get_importance(self):
		return sum(tree.get_importance() for tree in self.forest)/self.tree_num

	def fit(self, x, y):
		pred = np.zeros(y.shape)
		for i in range(self.tree_num):
			gradient = pred - y
			self.forest.append(XGBoostRegressionTree(max_depth=self.max_depth))
			self.forest[i].fit(x, gradient)
			pred += np.array([self.forest[i].predict(xi) * self.shrinkage for xi in x])
			print("tree {} constructed, loss {}".format(i, square_loss(pred, y)))

	def predict(self, x):
		return np.array([sum(tree.predict(xi) * self.shrinkage for tree in self.forest) for xi in x])


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

	xgboost = XGBoost()
	xgboost.fit(train_x, train_y)
	print(xgboost.get_importance())
	print(square_loss(test_y, xgboost.predict(test_x)))


if __name__ == "__main__":
	main()