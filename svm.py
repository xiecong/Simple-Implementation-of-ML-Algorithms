import numpy as np
from sklearn.datasets import load_digits
# work in rogress


class SVM(object):
    def __init__(self):
    	pass

    def fit(self, x, y):
    	pass

    def predict(self, x):
    	pass


def main():
	data = load_digits()
	test_ratio = 0.2
	test_split = np.random.uniform(0, 1, len(x))
	train_x = data.data[test_split >= test_ratio]
	test_x = data.data[test_split < test_ratio]
	train_y = data.target[test_split >= test_ratio]
	test_y = data.target[test_split < test_ratio]

	svm = SVM()
	svm.fit(train_x, train_y)
	print(sum(svm.predict(train_x) ==  train_y)/train_x.shape[0])
	print(sum(svm.predict(test_x) ==  test_y)/test_x.shape[0])


if __name__ == "__main__":
    main()