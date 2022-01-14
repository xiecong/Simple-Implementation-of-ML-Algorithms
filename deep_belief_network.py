import numpy as np
from sklearn.datasets import load_digits, fetch_openml
from multilayer_perceptron import MLP
from restricted_boltzmann_machine import RBM


def softmax(x):
    eps = 1e-8
    out = np.exp(x - np.max(x, axis=1).reshape(-1, 1))
    return out / (np.sum(out, axis=1).reshape(-1, 1) + eps)


# this implementation reused the training of MLP for back propagation
class DBN(object):

    def __init__(self, layers, n_labels):
        self.rbms = []
        self.n_labels = n_labels
        for n_v, n_h in zip(layers[:-1], layers[1:]):
            self.rbms.append(RBM(n_v, n_h, epochs=10, lr=0.1))
        self.dense = None

    def pretrain(self, x):
        v = x
        for rbm in self.rbms:
            rbm.fit(v)
            v = rbm.marginal_h(v)

    def finetuning(self, x, labels):
        # assign weights
        layers = [x.shape[1]] + [rbm.b.shape[1] for rbm in self.rbms] + [self.n_labels]
        mlp = MLP(act_type='Sigmoid', opt_type='Adam', layers=layers,
            epochs=20, learning_rate=0.01, lmbda=1e-2)
        
        mlp.w = [rbm.w for rbm in self.rbms] + \
            [np.random.randn(self.rbms[-1].w.shape[1], self.n_labels)]
        mlp.b = [rbm.b for rbm in self.rbms] + \
            [np.random.randn(1, self.n_labels)]
        mlp.fit(x, labels)
        # give back the weights
        # add the last feed-forward layer
        for rbm, w, b in zip(self.rbms, mlp.w[:-1], mlp.b[:-1]):
            rbm.w = w
            rbm.b = b
        self.dense = {'w': mlp.w[-1], 'b': mlp.b[-1]}

    def fit(self, x, y):
        self.pretrain(x)
        self.finetuning(x, y)

    def predict(self, x):
        for rbm in self.rbms:
            x = rbm.marginal_h(x)
        return softmax(self.dense['b'] + x.dot(self.dense['w']))


def main():
    # data = load_digits()
    # x, y = data.data, data.target
    x, y = fetch_openml('mnist_784', return_X_y=True, data_home="data")
    test_ratio = 0.2
    test_split = np.random.uniform(0, 1, x.shape[0])
    train_x, test_x = x[test_split >= test_ratio] / \
        x.max(), x[test_split < test_ratio] / x.max()
    train_y, test_y = y.astype(np.int_)[test_split >= test_ratio], y.astype(
        np.int_)[test_split < test_ratio]

    print('dbn training')
    dbn = DBN([train_x.shape[1], 100, 100], 10)
    dbn.fit(train_x, train_y)
    print('dbn train accuracy', sum(
        np.argmax(dbn.predict(train_x), axis=1) == train_y) / train_y.shape[0])
    print('dbn test accuracy', sum(
        np.argmax(dbn.predict(test_x), axis=1) == test_y) / test_y.shape[0])


if __name__ == "__main__":
    main()
