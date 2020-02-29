import numpy as np
from sklearn.datasets import load_digits


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression(object):

    def __init__(self):
        self.learning_rate = 0.01
        self.gamma = 0.9
        self.decay = 1 - 1e-4

    def loss(self, x, y):  # using cross entropy as loss function
        eps = 1e-20
        h = self.predict(x)
        return -(np.multiply(y, np.log(h + eps)) + np.multiply((1 - y), np.log(1 - h + eps))).mean()

    def fit(self, x, y):
        label_num = len(np.unique(y))
        labels = np.zeros((x.shape[0], label_num))
        labels[np.arange(x.shape[0]), y] = 1
        self.w = np.random.randn(x.shape[1], label_num)
        self.b = np.random.randn(1, label_num)
        self.mom_w = np.zeros_like(self.w)
        self.mom_b = np.zeros_like(self.b)

        train_num = x.shape[0]
        for i in range(5000):
            h = sigmoid(x.dot(self.w) + self.b)
            g_w = x.T.dot(h - labels) / train_num
            g_b = (h - labels).sum() / train_num
            self.mom_w = self.gamma * self.mom_w + self.learning_rate * g_w
            self.w = (self.w - self.mom_w) * self.decay
            self.mom_b = self.gamma * self.mom_b + self.learning_rate * g_b
            self.b = (self.b - self.mom_b) * self.decay
            if i % 100 == 0:
                print(self.loss(x, labels))

    def predict(self, x):
        return sigmoid(x.dot(self.w) + self.b)


def main():
    data = load_digits()
    test_ratio = 0.2
    test_split = np.random.uniform(0, 1, len(data.data))
    train_x, train_y = data.data[
        test_split >= test_ratio], data.target[test_split >= test_ratio]
    test_x, test_y = data.data[test_split < test_ratio], data.target[
        test_split < test_ratio]

    lr = LogisticRegression()
    lr.fit(train_x, train_y)
    print(sum(np.argmax(lr.predict(train_x), axis=1)
              == train_y) / train_y.shape[0])
    print(sum(np.argmax(lr.predict(test_x), axis=1)
              == test_y) / test_y.shape[0])


if __name__ == "__main__":
    main()
