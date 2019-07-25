import numpy as np
from sklearn.datasets import load_digits
from decision_tree import DecisionTree
# TODO output score/probability


class RandomForest(object):
	def __init__(self, tree_num=50, max_depth=5):
		self.max_depth = max_depth
		self.tree_num = tree_num
		self.forest = []

	def fit(self, x, y):
		feat_num = x.shape[1]
		n_feat = int(np.ceil(np.sqrt(feat_num)))
		data_num = x.shape[0]
		n_sample = data_num//5

		for i in range(self.tree_num):
			f = np.random.randint(feat_num, size=n_feat)
			idx = np.random.randint(data_num, size=n_sample)
			dt = DecisionTree(metric_type='Gini impurity', depth=self.max_depth)
			dt.fit(x[idx], y[idx], feature_set=f)
			self.forest.append(dt)
			print("Tree #{} constructed, acc {}".format(i, (self.predict(x)==y).sum()/x.shape[0]))

	def predict(self, x):
		preds = np.array([tree.predict(x) for tree in self.forest]).T
		value_counts = [np.unique(pred, return_counts=True) for pred in preds]
		return np.array([values[np.argmax(counts)] for (values, counts) in value_counts])


def main():
	data = load_digits()
	test_ratio = 0.2
	test_split = np.random.uniform(0, 1, len(x))
	train_x = data.data[test_split >= test_ratio]
	test_x = data.data[test_split < test_ratio]
	train_y = data.target[test_split >= test_ratio]
	test_y = data.target[test_split < test_ratio]

	rf = RandomForest()
	rf.fit(train_x, train_y)
	print((rf.predict(train_x) == train_y).sum()/train_x.shape[0])
	print((rf.predict(test_x) == test_y).sum()/test_x.shape[0])


if __name__ == "__main__":
    main()