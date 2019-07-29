import numpy as np
from sklearn.datasets import load_breast_cancer
from decision_tree import DecisionTree


class AdaBoost(object):
	def __init__(self, esti_num=10):
		self.esti_num = esti_num
		self.estimators = []
		self.alphas = []

	def fit(self, x, y):
		n_data = x.shape[0]
		w = np.ones(x.shape[0]) / n_data
		eps = 1e-16
		prediction = np.zeros(n_data)
		for i in range(self.esti_num):
			self.estimators.append(DecisionTree(metric_type='Gini impurity', depth=2))
			self.estimators[i].fit(x, y, w)
			pred_i = self.estimators[i].predict(x)
			error_i = (pred_i!=y).dot(w.T)
			self.alphas.append(np.log((1.0-error_i)/(error_i+eps))/2)
			w = w * np.exp(self.alphas[i]*(2*(pred_i!=y)-1))
			w = w / w.sum()

			prediction += pred_i * self.alphas[i]
			print("Tree {} constructed, acc {}".format(i, (np.sign(prediction) == y).sum()/n_data))

	def predict(self, x):
		return sum(esti.predict(x) * alpha for esti, alpha in zip(self.estimators, self.alphas))


def main():
	data = load_breast_cancer()
	y = data.target*2-1
	test_ratio = 0.2
	test_split = np.random.uniform(0, 1, len(data.data))
	train_x = data.data[test_split >= test_ratio]
	test_x = data.data[test_split < test_ratio]
	train_y = y[test_split >= test_ratio]
	test_y = y[test_split < test_ratio]

	adaboost = AdaBoost()
	adaboost.fit(train_x, train_y)
	print((np.sign(adaboost.predict(train_x))==train_y).sum()/train_x.shape[0])
	print((np.sign(adaboost.predict(test_x))==test_y).sum()/test_x.shape[0])


if __name__ == "__main__":
    main()