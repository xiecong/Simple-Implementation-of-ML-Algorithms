import numpy as np
from sklearn import datasets
from decision_tree import DecisionTree

def square_loss(y, pred):
	return np.square(pred-y).mean()/2

class XGBoostRegressionTree(DecisionTree):
	def __init__(self, data, max_depth):
		self.lambd = 0.01
		self.gamma = 0.1
		super(XGBoostRegressionTree, self).__init__(data=data, metric_type="Gini impurity", depth=max_depth, regression=True)
		self.metric = self.score

	def gen_leaf(self, data):
		return {'label': -data[:,-1].sum()/(data.shape[0]+self.lambd)}

	def score(self, data):
		return np.square(data[:,-1].sum())/(data.shape[0]+self.lambd)

	def split_gain(self, p_score, l_child, r_child):
		return self.metric(l_child) + self.metric(r_child) - p_score - self.gamma

# importance for each feature
class XGBoost(object):
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.max_depth = 5
		self.tree_num = 10
		self.forest = []
		self.shrinkage = 1

	def get_importance(self):
		return sum(tree.get_importance() for tree in self.forest)/self.tree_num

	def fit(self):
		pred = np.zeros(self.y.shape)
		for i in range(self.tree_num):
			gradient = pred - self.y
			self.forest.append(XGBoostRegressionTree(data=np.c_[self.x, gradient], max_depth=self.max_depth))
			self.forest[i].fit()
			pred += np.array([self.forest[i].predict(xi) * self.shrinkage for xi in self.x])
			print("tree {} constructed, loss {}".format(i, square_loss(pred,self.y)))

	def predict(self, x):
		return np.array([sum(tree.predict(xi) * self.shrinkage for tree in self.forest) for xi in x])

def main():
	data = datasets.load_boston()
	x = data.data
	y = data.target
	xgboost = XGBoost(x, y)
	xgboost.fit()
	print(xgboost.get_importance())

if __name__ == "__main__":
	main()