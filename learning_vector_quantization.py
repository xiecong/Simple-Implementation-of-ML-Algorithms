import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt


class LVQ(object):

    def __init__(self):
        self.lr = 0.01
        self.iterations = 10
        self.eps = 0.05
        self.w = None
        self.c = None

    def fit(self, x, y):
        n_labels = 10
        n_repeat = 10
        self.w = np.repeat([x[y==i].mean(axis=0) for i in range(n_labels)], n_repeat, axis=0)
        print(self.w.shape)
        self.c = np.repeat(np.arange(n_labels), n_repeat)
        for step in range(self.iterations):
            print(f'iteration {step}')
            for i in np.random.permutation(np.arange(x.shape[0])):
                j = np.argmin(np.square(self.w - x[i]).sum(axis=1))
                self.w[j] += (1.0 if self.c[j] == y[i] else -1.0) * self.lr * (x[i] - self.w[j])
            self.lr *= np.exp(-step * self.eps)

    def predict(self, x):
        return self.c[
            [np.argmin(np.square(self.w - xi).sum(axis=1)) for xi in x]
        ]


def main():
    x, y = fetch_openml('mnist_784', return_X_y=True, data_home="data", as_frame=False)
    test_ratio = 0.2
    test_split = np.random.uniform(0, 1, x.shape[0])
    train_x, test_x = x[test_split >= test_ratio] / \
        x.max(), x[test_split < test_ratio] / x.max()
    train_y, test_y = y.astype(np.int_)[test_split >= test_ratio], y.astype(
        np.int_)[test_split < test_ratio]

    lvq = LVQ()
    lvq.fit(train_x, train_y)
    print(sum(lvq.predict(train_x) == train_y) / train_y.shape[0])
    print(sum(lvq.predict(test_x) == test_y) / test_y.shape[0])

    for i in range(lvq.w.shape[0]):
        plt.subplot(10, 10, i+1)
        plt.imshow(lvq.w[i].reshape(28, 28), cmap='gray', vmin=np.min(lvq.w), vmax=np.max(lvq.w))
    plt.title('lvq codebooks')
    print('visualizing codebooks')
    plt.show()


if __name__ == "__main__":
    main()
