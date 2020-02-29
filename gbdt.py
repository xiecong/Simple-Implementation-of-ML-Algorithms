import numpy as np
from sklearn.datasets import fetch_california_housing
from decision_tree import DecisionTree
# TODO classification


def squared_loss(y, pred):
    return np.square(pred - y).mean() / 2


def squared_loss_gradient(y, pred):
    return pred - y


def absolute_loss_gradient(y, pred):
    return np.sign(pred - y)


class GBDT(object):

    def __init__(self, regression=True, tree_num=20, max_depth=4):
        self.regression = regression
        self.max_depth = max_depth
        self.tree_num = tree_num
        self.forest = []
        self.rhos = np.ones(self.tree_num)
        self.t0 = 0
        self.shrinkage = 0.5

    def get_importance(self):
        return sum(tree.get_importance() for tree in self.forest) / self.tree_num

    def _linear_search(self, y, pred, delta):
        step = 0.1
        rhos = np.arange(step, 10, step)
        losses = [squared_loss(y, pred - rho * delta) for rho in rhos]
        return rhos[np.argmin(losses)]

    def fit(self, x, y):
        self.t0 = y.mean()  # t0, which is a constant
        pred = y.mean()
        for i in range(self.tree_num):
            grad = squared_loss_gradient(y, pred)
            self.forest.append(DecisionTree(
                metric_type="Variance", depth=self.max_depth, regression=True))
            self.forest[i].fit(x, grad)
            delta = self.forest[i].predict(x)
            # find best learning rate
            self.rhos[i] = self._linear_search(y, pred, delta)
            pred -= self.shrinkage * delta * self.rhos[i]
            # for categorical dataset, use cross entropy loss
            print("tree {} constructed, rho {}, loss {}".format(
                i, self.rhos[i], squared_loss(y, pred)))

    def predict(self, x):
        return self.t0 - np.array([tree.predict(x) * rho * self.shrinkage for tree, rho in zip(self.forest, self.rhos)]).sum(axis=0)


def main():
    data = fetch_california_housing(data_home='data')
    test_ratio = 0.2
    test_split = np.random.uniform(0, 1, len(data.data))
    train_x = data.data[test_split >= test_ratio]
    test_x = data.data[test_split < test_ratio]
    train_y = data.target[test_split >= test_ratio]
    test_y = data.target[test_split < test_ratio]

    gbdt = GBDT()
    gbdt.fit(train_x, train_y)
    print(gbdt.get_importance())
    print(squared_loss(train_y, gbdt.predict(train_x)))
    print(squared_loss(test_y, gbdt.predict(test_x)))


if __name__ == "__main__":
    main()
