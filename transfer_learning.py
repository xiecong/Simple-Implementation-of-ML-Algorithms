import numpy as np
from sklearn.datasets import fetch_openml
from nn_layers import Conv, MaxPooling, FullyConnect, Activation, Softmax, BatchNormalization


class NN(object):

    def __init__(self, layers):
        self.layers = layers
        self.batch_size = 32
        self.epochs = 3

    def predict(self, x):
        out = x
        for layer in self.layers:
            out = layer.predict_forward(out) if isinstance(
                layer, BatchNormalization) else layer.forward(out)
        return out

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def gradient(self, grad_loss):
        grad = grad_loss
        for layer in self.layers[::-1]:
            grad = layer.gradient(grad)
        return grad

    def backward(self):
        for layer in self.layers:
            layer.backward()

    def predict(self, x):
        out = x
        for layer in self.layers:
            out = layer.predict_forward(out) if isinstance(
                layer, BatchNormalization) else layer.forward(out)
        return out


    def fit(self, x, labels):
        train_num = x.shape[0]
        n_labels = 5
        y = np.zeros((train_num, n_labels))
        y[np.arange(train_num), labels] = 1

        for epoch in range(self.epochs):
            permut = np.random.permutation(
                x.shape[0] // self.batch_size * self.batch_size).reshape([-1, self.batch_size])
            total_loss = 0
            count = 0
            for batch_idx in permut:
                pred = self.forward(x[batch_idx])
                loss = self.layers[-1].loss(pred, y[batch_idx])
                total_loss += loss

                if count % 100 == 0:
                    print("epoch {} batch {} loss: {}".format(
                        epoch, count, loss))
                count += 1

                # the last softmax layer calculates the pred - y
                self.gradient(y[batch_idx])  
                self.backward()
            print('avg batch loss', total_loss / permut.shape[0])


class TransferLearning(object):
    def __init__(self):
        self.lr = 0.001
        self.n_labels = 5

    def train(self, x, y):
        lr = self.lr
        conv1 = Conv(in_shape=x.shape[1:4], k_num=6, k_size=5, lr=lr)
        bn1 = BatchNormalization(in_shape=conv1.out_shape, lr=lr)
        relu1 = Activation(act_type="ReLU")
        pool1 = MaxPooling(in_shape=conv1.out_shape, k_size=2)
        conv2 = Conv(in_shape=pool1.out_shape, k_num=16, k_size=3, lr=lr)
        bn2 = BatchNormalization(in_shape=conv2.out_shape, lr=lr)
        relu2 = Activation(act_type="ReLU")
        pool2 = MaxPooling(in_shape=conv2.out_shape, k_size=2)
        fc = FullyConnect(pool2.out_shape, [self.n_labels], lr=lr)
        softmax = Softmax()

        nn = NN([
            conv1, bn1, relu1, pool1,
            conv2, bn2, relu2, pool2,
            fc, softmax
        ])
        nn.fit(x, y)
        return nn


    def transfer(self, x, y, nn):
        for layer in nn.layers[:-2]:
            x = layer.predict_forward(x) if isinstance(
                layer, BatchNormalization) else layer.forward(x)

        nn_top = NN([
            FullyConnect(nn.layers[-3].out_shape, [self.n_labels], lr=self.lr),
            Softmax()
        ])
        nn_top.fit(x, y)
        return NN(nn.layers[:-2] + nn_top.layers)


def main():
    x_all, y_all = fetch_openml('mnist_784', return_X_y=True, data_home="data")
    x_all = x_all.reshape(-1, 1, 28, 28)
    test_ratio = 0.2
    tl = TransferLearning()

    for mode_type in ['original', 'transferred']:
        index = (y_all <= '4') if mode_type == 'original' else (y_all > '4')
        x = x_all[index]
        y = y_all[index]
        test_split = np.random.uniform(0, 1, x.shape[0])
        train_x, train_y = x[test_split >= test_ratio] / \
            x.max(), y.astype(np.int_)[test_split >= test_ratio]
        test_x, test_y = x[test_split < test_ratio] / \
            x.max(), y.astype(np.int_)[test_split < test_ratio]
        if mode_type == 'original':
            print('train the first model')
            nn = tl.train(train_x, train_y)
        else:
            train_y = train_y - 5  # for one hot encoding purpose
            test_y = test_y - 5  # for one hot encoding purpose
            print('transfer to the second model')
            nn = tl.transfer(train_x, train_y, nn)
        print(nn.layers)
        print('model performance')
        print('train set accuracy', sum(np.argmax(nn.predict(train_x), axis=1) == train_y) / train_y.shape[0])
        print('test set accuracy', sum(np.argmax(nn.predict(test_x), axis=1) == test_y) / test_y.shape[0])


if __name__ == "__main__":
    main()
