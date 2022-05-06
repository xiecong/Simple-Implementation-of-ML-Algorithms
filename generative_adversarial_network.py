import numpy as np
from sklearn.datasets import fetch_openml
from nn_layers import FullyConnect, Activation, Softmax, BatchNormalization, Conv, TrasposedConv
import matplotlib.pyplot as plt


def noise(n_x, n_d):
    return np.random.randn(n_x, n_d)


def bce_loss(pred, y):
    eps = 1e-20
    return -((1 - y) * np.log(1 - pred + eps) + y * np.log(pred + eps)).mean()


def bce_grad(pred, y):
    eps = 1e-20
    return (- y / (pred + eps) + (1 - y) / (1 - pred + eps)) / pred.shape[0]


class NN(object):

    def __init__(self, layers):
        self.layers = layers

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


class GAN(object):

    def __init__(self, conditioned=True):
        self.n_epochs, self.batch_size = 1, 64
        self.gen_input = 100
        self.n_classes = 10
        self.conditioned = conditioned
        self.dc_gan()

    def dc_gan(self):
        gen_lr, dis_lr = 4e-3, 1e-3
        dense = FullyConnect(
            [self.gen_input + self.n_classes if self.conditioned else self.gen_input],
            (128, 7, 7), lr=gen_lr, optimizer='RMSProp'
        )
        tconv1 = TrasposedConv(dense.out_shape, k_size=4,
                               k_num=128, stride=2, padding=1, lr=gen_lr, optimizer='RMSProp')
        tconv2 = TrasposedConv(tconv1.out_shape, k_size=4,
                               k_num=128, stride=2, padding=1, lr=gen_lr, optimizer='RMSProp')
        tconv3 = TrasposedConv(tconv2.out_shape, k_size=7,
                               k_num=1, stride=1, padding=3, lr=gen_lr, optimizer='RMSProp')
        self.generator = NN([
            dense,
            BatchNormalization(tconv1.in_shape, lr=gen_lr, optimizer='RMSProp'),
            Activation(act_type='ReLU'),
            tconv1,
            BatchNormalization(tconv1.out_shape, lr=gen_lr, optimizer='RMSProp'),
            Activation(act_type='ReLU'),
            tconv2,
            BatchNormalization(tconv2.out_shape, lr=gen_lr, optimizer='RMSProp'),
            Activation(act_type='ReLU'),
            tconv3,
            BatchNormalization(tconv3.out_shape, lr=gen_lr, optimizer='RMSProp'),
            Activation(act_type='Tanh')
        ])
        conv1 = Conv(
            (1 + self.n_classes if self.conditioned else 1, 28, 28),
            k_size=7, k_num=128, stride=1, padding=3, lr=dis_lr, optimizer='RMSProp'
        )
        conv2 = Conv(conv1.out_shape, k_size=4, k_num=128,
                     stride=2, padding=1, lr=dis_lr, optimizer='RMSProp')
        conv3 = Conv(conv2.out_shape, k_size=4, k_num=128,
                     stride=2, padding=1, lr=dis_lr, optimizer='RMSProp')
        self.discriminator = NN([
            conv1,
            Activation(act_type='LeakyReLU'),
            conv2,
            BatchNormalization(conv2.out_shape, lr=dis_lr, optimizer='RMSProp'),
            Activation(act_type='LeakyReLU'),
            conv3,
            BatchNormalization(conv3.out_shape, lr=dis_lr, optimizer='RMSProp'),
            Activation(act_type='LeakyReLU'),
            FullyConnect(conv3.out_shape, [1], lr=dis_lr, optimizer='RMSProp'),
            Activation(act_type='Sigmoid')
        ])

    def fit(self, x, labels):
        y_true = np.ones((self.batch_size, 1))
        y_false = np.zeros((self.batch_size, 1))
        y_dis = np.concatenate([y_true, y_false], axis=0)
        label_channels = np.repeat(labels, 28*28, axis=1).reshape(labels.shape[0], self.n_classes, 28, 28)

        for epoch in range(self.n_epochs):
            permut = np.random.permutation(
                x.shape[0] // self.batch_size * self.batch_size).reshape([-1, self.batch_size])
            for b_idx in range(permut.shape[0]):
                batch_label_channels = label_channels[permut[b_idx, :]]
                if self.conditioned:
                    x_true = np.concatenate((x[permut[b_idx, :]], batch_label_channels), axis=1)
                else:
                    x_true = x[permut[b_idx, :]]
                pred_dis_true = self.discriminator.forward(x_true)
                self.discriminator.gradient(bce_grad(pred_dis_true, y_true))
                self.discriminator.backward()
                
                if self.conditioned:
                    x_gen = self.generator.forward(
                        np.concatenate((noise(self.batch_size, self.gen_input), labels[permut[b_idx, :]]), axis=1)
                    )
                    x_gen = np.concatenate((x_gen, batch_label_channels), axis=1)
                else:
                    x_gen = self.generator.forward(noise(self.batch_size, self.gen_input))
                pred_dis_gen = self.discriminator.forward(x_gen)
                self.discriminator.gradient(bce_grad(pred_dis_gen, y_false))
                self.discriminator.backward()

                pred_gen = self.discriminator.forward(x_gen)
                grad = self.discriminator.gradient(bce_grad(pred_gen, y_true))
                if self.conditioned:
                    self.generator.gradient(grad[:,:1,:,:])
                else: 
                    self.generator.gradient(grad)
                self.generator.backward()
                print(
                    f'Epoch {epoch} batch {b_idx} discriminator:',
                    bce_loss(np.concatenate((pred_dis_true, pred_dis_gen)), y_dis),
                    'generator:', bce_loss(pred_gen, y_true)
                )


def main():
    x, y = fetch_openml('mnist_784', return_X_y=True, data_home='data', as_frame=False)
    x = 2 * (x / x.max()) - 1
    labels = np.zeros((y.shape[0], 10))
    labels[range(y.shape[0]), y.astype(np.int_)] = 1
    gan = GAN(conditioned=True)
    gan.fit(x.reshape((-1, 1, 28, 28)), labels)

    if gan.conditioned:
        onehot = np.zeros((30, 10))
        onehot[range(30), np.arange(30)%10] = 1
        images = gan.generator.predict(
            np.concatenate((noise(30, gan.gen_input), onehot), axis=1)
        )
    else:
        images = gan.generator.predict(noise(30, gan.gen_input))

    for i, img in enumerate(np.array(images).reshape(-1, 784)):
        plt.subplot(len(images), 10, i + 1)
        plt.imshow(img.reshape(28, 28), cmap='gray', vmin=-1, vmax=1)
    plt.show()

if __name__ == "__main__":
    main()
