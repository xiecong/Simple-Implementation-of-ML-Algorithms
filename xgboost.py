import numpy as np
from sklearn.datasets import fetch_california_housing
from decision_tree import DecisionTree
# TODO classification


def squared_loss(y, pred):
    return np.square(pred - y).mean() / 2


def squared_loss_gradient(y, pred):
    return pred - y


class XGBoostRegressionTree(DecisionTree):

    def __init__(self, max_depth):
        self.lambd = 0.01
        self.gamma = 0.1
        super(XGBoostRegressionTree, self).__init__(
            metric_type="Gini impurity", depth=max_depth, regression=True)
        self.metric = self.score

    def gen_leaf(self, y, w):
        return {'label': y.dot(w) / (sum(w) + self.lambd)}

    def score(self, y, w):
        return np.square(y.dot(w)) / (sum(w) + self.lambd)

    def split_gain(self, p_score, l_y, r_y, l_w, r_w):
        return (self.metric(l_y, l_w) + self.metric(r_y, r_w) - p_score) / 2 - self.gamma

# importance for each feature


class XGBoost(object):

    def __init__(self, regression=True, max_depth=4, tree_num=20):
        self.regression = regression
        self.max_depth = max_depth
        self.tree_num = tree_num
        self.forest = []
        self.shrinkage = 0.5

    def get_importance(self):
        return sum(tree.get_importance() for tree in self.forest) / self.tree_num

    def fit(self, x, y):
        pred = 0
        for i in range(self.tree_num):
            grad = squared_loss_gradient(y, pred)
            self.forest.append(XGBoostRegressionTree(max_depth=self.max_depth))
            self.forest[i].fit(x, grad)
            pred -= self.forest[i].predict(x) * self.shrinkage
            print("tree {} constructed, loss {}".format(
                i, squared_loss(y, pred)))

    def predict(self, x):
        return -np.array([tree.predict(x) * self.shrinkage for tree in self.forest]).sum(axis=0)


def main():
    data = fetch_california_housing(data_home='data')
    test_ratio = 0.2
    test_split = np.random.uniform(0, 1, len(data.data))
    train_x = data.data[test_split >= test_ratio]
    test_x = data.data[test_split < test_ratio]
    train_y = data.target[test_split >= test_ratio]
    test_y = data.target[test_split < test_ratio]

    xgboost = XGBoost()
    xgboost.fit(train_x, train_y)
    print(xgboost.get_importance())
    print(squared_loss(train_y, xgboost.predict(train_x)))
    print(squared_loss(test_y, xgboost.predict(test_x)))


if __name__ == "__main__":
    main()
