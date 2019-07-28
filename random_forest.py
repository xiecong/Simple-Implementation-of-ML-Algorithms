import numpy as np
from sklearn.datasets import load_digits
from decision_tree import DecisionTree


class RandomForest(object):
	def __init__(self, tree_num=50, max_depth=5, regression=False):
		self.max_depth = max_depth
		self.tree_num = tree_num
		self.forest = []
		self.regression = regression
		self.metric_type = 'Variance' if regression else 'Gini impurity'

	def fit(self, x, y):
		feat_num = x.shape[1]
		n_feat = int(np.sqrt(feat_num))
		data_num = x.shape[0]
		n_sample = data_num//5
		self.labels = np.unique(y)

		for i in range(self.tree_num):
			f = np.random.randint(feat_num, size=n_feat)
			idx = np.random.randint(data_num, size=n_sample)
			dt = DecisionTree(metric_type=self.metric_type, depth=self.max_depth, regression=self.regression)
			dt.fit(x[idx], y[idx], feature_set=f)
			self.forest.append(dt)
			if self.regression:
				print("Tree #{} constructed, squared loss {}".format(i, np.square(self.predict(x)-y).sum()))
			else:
				print("Tree #{} constructed, acc {}".format(i, (np.argmax(self.predict(x), axis=1)==y).sum()/x.shape[0]))

	def predict(self, x):
		preds = np.array([tree.predict(x) for tree in self.forest]).T
		if self.regression:
			return preds.mean(axis=1)
		else:
			y = np.zeros((x.shape[0], len(self.labels)))
			for i, pred in enumerate(preds):
				value, counts = np.unique(pred, return_counts=True)
				y[i][value.astype(int)] = counts / counts.sum()
			return y


def main():
	data = load_digits()
	test_ratio = 0.2
	test_split = np.random.uniform(0, 1, len(data.data))
	train_x = data.data[test_split >= test_ratio]
	test_x = data.data[test_split < test_ratio]
	train_y = data.target[test_split >= test_ratio]
	test_y = data.target[test_split < test_ratio]

	rf = RandomForest()
	rf.fit(train_x, train_y)
	print((np.argmax(rf.predict(train_x), axis=1) == train_y).sum()/train_x.shape[0])
	print((np.argmax(rf.predict(test_x), axis=1) == test_y).sum()/test_x.shape[0])


if __name__ == "__main__":
    main()