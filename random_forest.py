import numpy as np
from sklearn.datasets import load_digits
from decision_tree import DecisionTree


class RandomForest(object):
	def __init__(self):
		self.max_depth = 5
		self.tree_num = 20
		self.forest = []

	def fit(self, x, y):
		feat_num = x.shape[1]
		n_feat = int(np.sqrt(feat_num))
		data_num = x.shape[0]
		n_sample = data_num//5

		for i in range(self.tree_num):
			print("fitting tree #{}".format(i))
			f = np.random.randint(feat_num, size=n_feat)
			idx = np.random.randint(data_num, size=n_sample)
			dt = DecisionTree('Gini impurity', self.max_depth)
			dt.fit(x[idx], y[idx])
			self.forest.append(dt)

	def predict(self, x):
		res = [tree.predict(x) for tree in self.forest]
		(values, counts) = np.unique(res, return_counts=True)
		return values[np.argmax(counts)]#dict(zip(values, counts/counts.sum()))


def main():
	data = load_digits()
	x = data.data
	y = data.target

	test_ratio = 0.2
	test_split = np.random.uniform(0, 1, len(x))
	train_x = x[test_split >= test_ratio]
	test_x = x[test_split < test_ratio]
	train_y = y[test_split >= test_ratio]
	test_y = y[test_split < test_ratio]

	rf = RandomForest()
	rf.fit(train_x, train_y)
	print(sum(rf.predict(xi) == yi for xi, yi in zip(test_x, test_y))/test_x.shape[0])


if __name__ == "__main__":
    main()