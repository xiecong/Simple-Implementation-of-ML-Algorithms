import numpy as np
from sklearn.datasets import load_digits
## a simpler implementation of multilayer perceptron with backpropagation training
## 2 hidden layers, with 100 and 50 perceptrons
## this one use sigmoid in hiddn layer and softmax in output
## set batch size and epochs before start


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    eps = 1e-8
    out = np.exp(x - np.max(x, axis=1).reshape(-1, 1))
    return out / (np.sum(out, axis=1).reshape(-1, 1) + eps)


class MLP(object):
    def __init__(self, n_features, n_labels):
        '''
        D_in is input dimension;
        H is hidden dimension; 
        D_out is output dimension.
        '''
        self.D_in, self.H1, self.H2, self.D_out = n_features, 100, 50, n_labels
        self.epochs, self.batch_size = 200, 32
        self.learning_rate = 1e-2

        # Randomly initialize weights
        self.w1 = np.random.randn(self.D_in, self.H1)
        self.w2 = np.random.randn(self.H1, self.H2)
        self.w3 = np.random.randn(self.H2, self.D_out)

        self.b1 = np.random.randn(1, self.H1)
        self.b2 = np.random.randn(1, self.H2)
        self.b3 = np.random.randn(1, self.D_out)
    
    def loss(self, x, y):
        return -(np.multiply(y, np.log(self.predict(x)))).mean()

    def predict(self, x):
        eps = 1e-8
        a1 = sigmoid(x.dot(self.w1) + self.b1)
        a2 = sigmoid(a1.dot(self.w2) + self.b2)
        return softmax(a2.dot(self.w3) + self.b3)

    def fit(self, x_train, labels):
        train_num = x_train.shape[0]
        eps = 1e-8
        bvec = np.ones((1, self.batch_size))
        
        y_train = np.zeros((train_num, self.D_out))
        y_train[np.arange(train_num), labels] = 1

        for epoch in range(self.epochs):
            #mini batch
            permut=np.random.permutation(train_num//self.batch_size*self.batch_size).reshape(-1,self.batch_size)
            for b_idx in range(permut.shape[0]):
                x, y = x_train[permut[b_idx,:]], y_train[permut[b_idx,:]]
                
                # Forward pass: compute predicted y
                a1 = sigmoid(x.dot(self.w1) + self.b1)
                a2 = sigmoid(a1.dot(self.w2) + self.b2)
                out = softmax(a2.dot(self.w3) + self.b3)

                # Backprop to compute gradients of weights with respect to loss
                grad_out = out - y
                grad_w3 = a2.T.dot(grad_out)

                grad_a2 = grad_out.dot(self.w3.T)
                grad_a2 = np.multiply(grad_a2, (a2 - np.square(a2)))
                grad_w2 = a1.T.dot(grad_a2)

                grad_a1 = grad_a2.dot(self.w2.T)
                grad_a1 = np.multiply(grad_a1, (a1 - np.square(a1)))
                grad_w1 = x.T.dot(grad_a1)

                # Update weights
                self.w1 -= self.learning_rate * grad_w1
                self.b1 -= self.learning_rate * bvec.dot(grad_a1)
                self.w2 -= self.learning_rate * grad_w2
                self.b2 -= self.learning_rate * bvec.dot(grad_a2)
                self.w3 -= self.learning_rate * grad_w3
                self.b3 -= self.learning_rate * bvec.dot(grad_out)
            print('epoch {}, loss: {}'.format(epoch, self.loss(x_train, y_train)))


def main():
    data = load_digits()
    test_ratio = 0.2
    test_split = np.random.uniform(0, 1, len(data.data))
    train_x = data.data[test_split >= test_ratio] / data.data.max()
    test_x = data.data[test_split < test_ratio] / data.data.max()
    train_y = data.target[test_split >= test_ratio]
    test_y = data.target[test_split < test_ratio]

    mlp = MLP(train_x.shape[1], len(np.unique(data.target)) )
    mlp.fit(train_x, train_y)
    print(sum(np.argmax(mlp.predict(train_x), axis=1) == train_y)/train_y.shape[0])
    print(sum(np.argmax(mlp.predict(test_x), axis=1) == test_y)/test_y.shape[0])


if __name__ == "__main__":
    main()