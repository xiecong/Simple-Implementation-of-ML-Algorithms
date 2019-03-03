import numpy as np
from sklearn import datasets
from decision_tree import DecisionTree

def square_loss(y, pred):
	return np.square(pred-y).mean()/2

def gradient(y, pred):
	return pred - y

# will add feature importance calculation
class GBDT(object):
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.max_depth = 5
		self.tree_num = 10
		self.forest = []
		self.rhos = []
		# tree0 is a constant
		self.t0 = np.ones(self.y.shape[0]) * self.y.mean()

	def get_importance(self):
		return sum(tree.get_importance() for tree in self.forest)/self.tree_num

	def linear_search(self, y, pred, f):
		shrinkage = 1	
		losses = [square_loss(y,pred+i*f/10) for i in range(1,21)]
		return shrinkage * (np.argmin(losses)+1)/10

	def fit(self):
		pred = self.t0
		for i in range(self.tree_num):
			grad = gradient(self.y, pred)
			self.forest.append(DecisionTree(data=np.c_[self.x, grad],
											metric_type="Variance",
											depth=self.max_depth,
											regression=True))
			self.forest[i].fit()
			f = np.array([self.forest[i].predict(xi) for xi in self.x])
			# find best learning rate
			self.rhos.append(self.linear_search(self.y, pred, f))
			pred -= f * self.rhos[i]
			# for categorical dataset, use cross entropy loss
			print("tree {} constructed, loss {}".format(i, square_loss(pred,self.y)))

	def predict(self, x):
		return self.t0 - sum(np.array([tree.predict(xi)for xi in x]) * rho for tree, rho in zip(self.forest, self.rhos))

def main():
	data = datasets.load_boston()
	x = data.data
	y = data.target
	gbdt = GBDT(x, y)
	gbdt.fit()
	print(gbdt.get_importance())
	print(square_loss(gbdt.predict(x), y))
	
if __name__ == "__main__":
	main()