import numpy as np
import matplotlib.pyplot as plt
## a simpler implementation of multilayer perceptron with backpropagation training
## 3 hidden layers, each with 8 perceptrons
## this one only use ReLU
## set batch size and epochs before start

def gen_xor_data(train_num, dim_in):
    x = 2 * np.random.random((train_num, dim_in)) - 1
    y = np.array([[1] if(xi[0]*xi[1]>0) else [-1] for xi in x])
    return x, y

# visualize decision boundary change
def boundary_vis(mlp):
    clabel = ['red' if yi[0] < 0 else 'blue' for yi in mlp.y]
    xx, yy = np.meshgrid(np.linspace(-1, 1, 50), np.linspace(-1, 1, 50))
    zz = mlp.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    loss = np.square(mlp.predict(mlp.x) - mlp.y).sum() / 2
    plt.title("Decision Boundary (epoch {}, loss={})".format(mlp.epochs, loss))
    plt.contourf(xx, yy, zz, levels=np.linspace(zz.min(), zz.max(), 40), cmap=plt.cm.RdBu)
    plt.contour(xx, yy, zz, levels=[0], colors='darkred')
    plt.scatter(mlp.x[:, 0], mlp.x[:, 1], c=clabel, s=10, edgecolors='k')
    plt.show()

class MLP():
    def __init__(self):
        '''
        D_in is input dimension;
        H is hidden dimension; 
        D_out is output dimension.
        '''
        self.D_in, self.H1, self.H2, self.D_out = 2, 6, 6, 1
        self.train_num = 300
        self.epochs = 400
        self.batch_size = 10 
        self.learning_rate = 1e-3

        self.x, self.y = gen_xor_data(self.train_num, self.D_in)
        # Randomly initialize weights
        self.w1 = np.random.randn(self.D_in, self.H1)
        self.w2 = np.random.randn(self.H1, self.H2)
        self.w3 = np.random.randn(self.H2, self.D_out)

        self.b1 = np.random.randn(1, self.H1)
        self.b2 = np.random.randn(1, self.H2)
        self.b3 = np.random.randn(1, self.D_out)

    def predict(self, x):
        a1 = np.maximum(x.dot(self.w1) + self.b1, 0)
        a2 = np.maximum(a1.dot(self.w2) + self.b2, 0)
        return a2.dot(self.w3) + self.b3

    def train(self):
        bvec = np.ones((1, self.batch_size))
        for epoch in range(self.epochs):
            #mini batch
            permut=np.random.permutation(self.train_num).reshape(-1,self.batch_size)
            for b_idx in range(permut.shape[0]):
                x, y = self.x[permut[b_idx,:]], self.y[permut[b_idx,:]]
                
                # Forward pass: compute predicted y
                a1 = np.maximum(x.dot(self.w1) + self.b1, 0)
                a2 = np.maximum(a1.dot(self.w2) + self.b2, 0)
                out = a2.dot(self.w3) + self.b3

                # Backprop to compute gradients of weights with respect to loss
                grad_out = out - y
                grad_w3 = a2.T.dot(grad_out)

                grad_a2 = grad_out.dot(self.w3.T)
                grad_a2[a2 <= 0] = 0
                grad_w2 = a1.T.dot(grad_a2)

                grad_a1 = grad_a2.dot(self.w2.T)
                grad_a1[a1 <= 0] = 0
                grad_w1 = x.T.dot(grad_a1)

                # Update weights
                self.w1 -= self.learning_rate * grad_w1
                self.b1 -= self.learning_rate * bvec.dot(grad_a1)
                self.w2 -= self.learning_rate * grad_w2
                self.b2 -= self.learning_rate * bvec.dot(grad_a2)
                self.w3 -= self.learning_rate * grad_w3
                self.b3 -= self.learning_rate * bvec.dot(grad_out)

mlp = MLP()
mlp.train()
boundary_vis(mlp)