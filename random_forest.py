import numpy as np
from decision_tree import DecisionTree

class RandomForest():
	def __init__(self, data):
		self.max_depth = 3
		self.tree_num = 10
		self.n_feat = 7
		self.forest = []
		self.data = data
		self.feat_num = data.shape[1] - 1
		self.data_num = data.shape[0]
		self.n_sample = 40

	def train(self):
		for i in range(self.tree_num):
			f = np.random.randint(self.feat_num, size=self.n_feat)
			idx = np.random.randint(self.data_num, size=self.n_sample)
			dt = DecisionTree(self.data[idx], 'Gini impurity', self.max_depth)
			dt.train()
			self.forest.append(dt)

	def predict(self, x):
		res = []
		for i in range(self.tree_num):
			res.append(self.forest[i].predict(x))
		(values, counts) = np.unique(res,return_counts=True)
		return values[np.argmax(counts)]#dict(zip(values, counts/counts.sum()))

def main():
	x = np.loadtxt("data/sonar.all-data.txt", delimiter = ',', usecols = range(60), dtype = float)
	y = np.loadtxt("data/sonar.all-data.txt", delimiter = ',', usecols = (60), dtype = str)
	rf = RandomForest(np.c_[x,y])
	rf.train()
	correct = 0
	for i, xi in enumerate(x):
		if(rf.predict(xi) == y[i]):
			correct+=1
	print(correct/x.shape[0])

if __name__ == "__main__":
    main()