import numpy as np
from sklearn import datasets
from decision_tree import DecisionTree

class XGBoostRegressionTree(DecisionTree):
	def __init__(self, data):
		self.max_depth = 8
		super(XGBoostRegressionTree, self).__init__(data, "", depth=self.max_depth, metric_type=None, metric_func=, regression=True)


# importance for each feature
class XGBoost(object):
	def __init__(self, data):
		self.x = data[:,:-1]
		self.y = data[:,-1]
		self.max_depth = 8
		self.tree_num = 10
		self.forest = []

	def fit(self):
		alpha = 1
		pred = np.zeros(self.y.shape)
		for i in range(self.tree_num):
			gradient = 
			dt = XGBoostRegressionTree(data=np.c_[self.x, gradient])
			self.forest.append(dt)
			pred += [dt.predict(xi) * alpha for xi in self.x]

	def predict(self, x):
		return [sum(tree.predict(xi) for tree in self.forest) for xi in x]

def main():
	data = datasets.load_boston()
	x = data.data
	y = data.target
	print(x.shape, y.shape)
	xgboost = XGBoost(x, y)

if __name__ == "__main__":
	main()