import numpy as np
from sklearn.datasets import fetch_openml
from nn_layers import Conv, MaxPooling, FullyConnect, Activation, Softmax, BatchNormalization
# This implements Lenet-4, test on MNIST dataset
# gradient check for all layers for input x, w, b


class CNN(object):

    def __init__(self, x_shape, label_num):
        self.batch_size, lr = 32, 1e-3
        # Conv > Normalization > Activation > Dropout > Pooling
        conv1 = Conv(in_shape=x_shape, k_num=6, k_size=5, lr=lr)
        bn1 = BatchNormalization(in_shape=conv1.out_shape, lr=lr)
        relu1 = Activation(act_type="ReLU")
        pool1 = MaxPooling(in_shape=conv1.out_shape, k_size=2)
        conv2 = Conv(in_shape=pool1.out_shape, k_num=16, k_size=3, lr=lr)
        bn2 = BatchNormalization(in_shape=conv2.out_shape, lr=lr)
        relu2 = Activation(act_type="ReLU")
        pool2 = MaxPooling(in_shape=conv2.out_shape, k_size=2)
        fc1 = FullyConnect(pool2.out_shape, [120], lr=lr)
        bn3 = BatchNormalization(in_shape=[120], lr=lr)
        relu3 = Activation(act_type="ReLU")
        fc2 = FullyConnect([120], [label_num], lr=lr)
        softmax = Softmax()

        self.layers = [
            conv1, bn1, relu1, pool1,
            conv2, bn2, relu2, pool2,
            fc1, bn3, relu3,
            fc2, softmax
        ]

    def fit(self, train_x, labels):
        n_data = train_x.shape[0]
        train_y = np.zeros((n_data, 10))
        train_y[np.arange(n_data), labels] = 1
        for epoch in range(3):
            # mini batch
            permut = np.random.permutation(
                n_data // self.batch_size * self.batch_size).reshape([-1, self.batch_size])
            total_loss = 0
            for b_idx in range(permut.shape[0]):
                x0 = train_x[permut[b_idx, :]]
                y = train_y[permut[b_idx, :]]

                out = x0
                for layer in self.layers:
                    out = layer.forward(out)

                batch_loss = self.layers[-1].loss(out, y)
                if b_idx % 100 == 0:
                    print("epoch {} batch {} loss: {}".format(
                        epoch, b_idx, batch_loss))
                grad = y
                for layer in self.layers[::-1]:
                    grad = layer.gradient(grad)
                for layer in self.layers:
                    layer.backward()
                total_loss += batch_loss
            print('acc', self.get_accuracy(train_x, labels),
                  'avg batch loss', total_loss / permut.shape[0])

    def predict(self, x):
        out = x
        for layer in self.layers:
            out = layer.predict_forward(out) if isinstance(
                layer, BatchNormalization) else layer.forward(out)
        return out

    def get_accuracy(self, x, label):
        n_correct = 0
        for i in range(0, x.shape[0], self.batch_size):
            x_batch, label_batch = x[
                i: i + self.batch_size], label[i: i + self.batch_size]
            n_correct += sum(np.argmax(self.predict(x_batch),
                                       axis=1) == label_batch)
        return n_correct / x.shape[0]


def gradient_check(conv=True):
    if conv:
        layera = Conv(in_shape=[16, 32, 28], k_num=12, k_size=3)
        layerb = Conv(in_shape=[16, 32, 28], k_num=12, k_size=3)
    else:
        layera = FullyConnect(in_shape=[16, 32, 28], out_dim=12)
        layerb = FullyConnect(in_shape=[16, 32, 28], out_dim=12)
    act_layer = Activation(act_type='Tanh')
    layerb.w = layera.w.copy()
    layerb.b = layera.b.copy()
    eps = 1e-4
    x = np.random.randn(10, 16, 32, 28) * 10
    for i in range(100):
        idxes = tuple((np.random.uniform(0, 1, 4) * x.shape).astype(int))
        x_a = x.copy()
        x_b = x.copy()
        x_a[idxes] += eps
        x_b[idxes] -= eps
        out = act_layer.forward(layera.forward(x))
        gradient = layera.gradient(act_layer.gradient(np.ones(out.shape)))

        delta_out = (act_layer.forward(layera.forward(x_a)) -
                     act_layer.forward(layerb.forward(x_b))).sum()
        # the output should be in the order of eps*eps
        print(idxes, (delta_out / eps / 2 - gradient[idxes]) / eps / eps)


def main():
    x, y = fetch_openml('mnist_784', return_X_y=True, data_home="data")
    x = x.reshape(-1, 1, 28, 28)

    test_ratio = 0.2
    test_split = np.random.uniform(0, 1, x.shape[0])
    train_x, train_y = x[test_split >= test_ratio] / \
        x.max(), y.astype(np.int_)[test_split >= test_ratio]
    test_x, test_y = x[test_split < test_ratio] / \
        x.max(), y.astype(np.int_)[test_split < test_ratio]

    cnn = CNN(x.shape[1:4], 10)
    cnn.fit(train_x, train_y)
    print('train acc', cnn.get_accuracy(train_x, train_y))
    print('test acc', cnn.get_accuracy(test_x, test_y))

if __name__ == "__main__":
    # gradient_check()
    main()
