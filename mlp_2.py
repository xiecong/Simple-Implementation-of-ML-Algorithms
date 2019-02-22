import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

## a simpler implementation of multilayer perceptron with backpropagation training
## 3 hidden layers, each with 8 perceptrons
## this one only use ReLU
## set batch size and epochs before start

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    eps = 1e-8
    out = np.exp(x - np.max(x, axis=1).reshape(-1, 1))
    return out / (np.sum(out, axis=1).reshape(-1, 1) + eps)

class MLP():
    def __init__(self, x, y):
        '''
        D_in is input dimension;
        H is hidden dimension; 
        D_out is output dimension.
        '''
        self.label_num = len(np.unique(y))  
        self.D_in, self.H1, self.H2, self.D_out = x.shape[1], 32, 32, self.label_num
        self.train_num = x.shape[0]
        self.epochs = 300
        self.batch_size = 15
        self.learning_rate = 1e-2

        self.x = x 
        self.y = np.zeros((self.train_num, self.label_num))
        self.y[np.arange(self.train_num), y] = 1

        # Randomly initialize weights
        self.w1 = np.random.randn(self.D_in, self.H1)
        self.w2 = np.random.randn(self.H1, self.H2)
        self.w3 = np.random.randn(self.H2, self.D_out)

        self.b1 = np.random.randn(1, self.H1)
        self.b2 = np.random.randn(1, self.H2)
        self.b3 = np.random.randn(1, self.D_out)
    
    def loss(self, x):
        return -(np.multiply(self.y, np.log(self.predict(x)))).mean()

    def predict(self, x):
        eps = 1e-8
        a1 = sigmoid(x.dot(self.w1) + self.b1)
        a2 = sigmoid(a1.dot(self.w2) + self.b2)
        return softmax(a2.dot(self.w3) + self.b3)

    def fit(self):
        eps = 1e-8
        bvec = np.ones((1, self.batch_size))
        for epoch in range(self.epochs):
            #mini batch
            permut=np.random.permutation(self.train_num//self.batch_size*self.batch_size).reshape(-1,self.batch_size)
            for b_idx in range(permut.shape[0]):
                x, y = self.x[permut[b_idx,:]], self.y[permut[b_idx,:]]
                
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
            #print(self.loss(self.x))

def main():
    data = datasets.load_digits()
    x = data.data
    y = data.target
    mlp = MLP(x, y)
    mlp.fit()   
    res = mlp.predict(mlp.x)
    print(sum(y[i]==np.argmax(o) for i, o in enumerate(res))/y.shape[0])

if __name__ == "__main__":
    main()