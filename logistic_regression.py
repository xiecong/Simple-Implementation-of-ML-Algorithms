import numpy as np
from sklearn.datasets import load_digits


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression(object):
	def __init__(self):
		pass

	def loss(self, x, y): #using cross entropy as loss function
		eps = 1e-8
		h = self.predict(x)
		return -(np.multiply(y, np.log(h+eps)) + np.multiply((1 - y), np.log(1 - h+eps))).mean()

	def fit(self, x, y):
		label_num = len(np.unique(y))	
		labels = np.zeros((x.shape[0], label_num))
		labels[np.arange(x.shape[0]), y] = 1
		self.w = np.random.randn(x.shape[1], label_num)
		self.b = np.random.randn(1, label_num)

		train_num = x.shape[0]
		learning_rate = 0.01
		for i in range(5000):
			h = sigmoid(x.dot(self.w)+self.b)
			g_w = x.T.dot(h - labels) / train_num
			g_b = (h - labels).sum() / train_num
			self.w -= learning_rate * g_w
			self.b -= learning_rate * g_b

	def predict(self, x):
		return sigmoid(x.dot(self.w) + self.b)


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

	lr = LogisticRegression()
	lr.fit(train_x, train_y)
	res = lr.predict(test_x)
	print(sum(yi==np.argmax(y_hat) for y_hat, yi in zip(res, test_y))/test_y.shape[0])


if __name__ == "__main__":
    main()