import numpy as np
from sklearn import datasets
from decision_tree import DecisionTree

class RandomForest(object):
	def __init__(self, data):
		self.max_depth = 8
		self.tree_num = 10
		self.forest = []
		self.data = data
		self.feat_num = data.shape[1] - 1
		self.n_feat = int(np.sqrt(self.feat_num))
		self.data_num = data.shape[0]
		self.n_sample = self.data_num//5

	def fit(self):
		for i in range(self.tree_num):
			print("fitting tree #{}".format(i))
			f = np.random.randint(self.feat_num, size=self.n_feat)
			idx = np.random.randint(self.data_num, size=self.n_sample)
			dt = DecisionTree(self.data[idx], 'Gini impurity', self.max_depth)
			dt.fit()
			self.forest.append(dt)

	def predict(self, x):
		res = [tree.predict(x) for tree in self.forest]
		(values, counts) = np.unique(res, return_counts=True)
		return values[np.argmax(counts)]#dict(zip(values, counts/counts.sum()))

def main():
	data = datasets.load_digits()
	x = data.data
	y = data.target
	rf = RandomForest(np.c_[x,y])
	rf.fit()
	print(sum(rf.predict(xi) == y[i] for i, xi in enumerate(x))/x.shape[0])

if __name__ == "__main__":
    main()