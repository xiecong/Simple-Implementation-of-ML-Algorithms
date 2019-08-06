import numpy as np
from sklearn.datasets import load_digits


class kNearestNeighbor(object):
	def __init__(self, k):
		self.k = k

	def fit(self, x, y):
		self.train_x = x
		self.train_y = y
		self.labels = np.unique(y)

	def _get_nn(self, x):
		nn_idx = np.argsort(np.square(self.train_x - x).sum(axis=1))[:self.k]
		nn_y, counts = np.unique(self.train_y[nn_idx], return_counts=True)
		y = np.zeros(len(self.labels))
		y[nn_y] = counts
		return y / y.sum()

	def predict(self, x):
		return np.array([self._get_nn(xi) for xi in x])


def main():
	data = load_digits()
	test_ratio = 0.2
	test_split = np.random.uniform(0, 1, len(data.data))
	train_x, test_x = data.data[test_split >= test_ratio], data.data[test_split < test_ratio]
	train_y, test_y = data.target[test_split >= test_ratio], data.target[test_split < test_ratio]

	knn = kNearestNeighbor(k=3)
	knn.fit(train_x, train_y)
	print(sum(np.argmax(knn.predict(train_x), axis=1)==train_y)/train_y.shape[0])
	print(sum(np.argmax(knn.predict(test_x), axis=1)==test_y)/test_y.shape[0])


if __name__ == "__main__":
    main()