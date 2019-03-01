import numpy as np
from sklearn import datasets
from decision_tree import DecisionTree

# will add feature importance calculation
class GBDT(object):
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.max_depth = 5
		self.tree_num = 20
		self.forest = []
		self.rhos = []
		# tree0 is a constant
		self.t0 = np.ones(self.y.shape[0]) * self.y.mean()

	def linear_search(self, y, pred, f):
		min_loss = np.float('inf')
		rho = 0
		for i in range(21):
			r = i/20
			loss = np.square(y-pred+r*f).mean()
			if(loss<min_loss):
				min_loss = loss
				rho = r
		return rho

	def fit(self):
		pred = self.t0
		for i in range(self.tree_num):
			gradient = pred - self.y 
			self.forest.append(DecisionTree(data=np.c_[self.x, gradient],
											metric_type="Variance",
											depth=self.max_depth,
											regression=True))
			self.forest[i].fit()
			f = np.array([self.forest[i].predict(xi) for xi in self.x])
			# find best learning rate
			self.rhos.append(self.linear_search(self.y, pred, f))
			pred -= f * self.rhos[i]
			# gradient for square loss
			# for categorical dataset, use cross entropy loss
			print("tree {} constructed, loss {}".format(i, np.square(pred-self.y).mean()/2))

	def predict(self, x):
		pred = self.t0
		for i in range(self.tree_num):
			pred -= np.array([self.forest[i].predict(xi) * self.rhos[i] for xi in x])
		return pred

def main():
	data = datasets.load_boston()
	x = data.data
	y = data.target
	gbdt = GBDT(x, y)
	gbdt.fit()
	#print(np.c_[gbdt.predict(x), y])

if __name__ == "__main__":
	main()
