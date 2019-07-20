import numpy as np
from sklearn.datasets import load_digits


class SVM(object):
    def __init__(self):
    	pass

    def fit(self, x, y):
    	pass

    def predict(self, x):
    	pass


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

	svm = SVM()
	svm.fit(train_x, train_y)
	print(sum(rf.predict(xi) == yi for xi, yi in zip(test_x, test_y))/test_x.shape[0])


if __name__ == "__main__":
    main()