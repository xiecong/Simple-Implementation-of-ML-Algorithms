import numpy as np
from decision_boundary_vis import gen_spiral_data, evol_boundary_vis
'''
This is a simple implementation of multilayer perceptron with backpropagation training
Implemented features:
Activation function: relu/sigmoid/hyperbolic tangent
Regularization: optional L1/L2 to prevent overfitting
Optimization: SDG/Momentum/Adagrad/RMSprop/Nesterov/Adam
Architecture: layer configuarations in self.layers
Hyperparameters: set learning rate, batch size and epochs before start
'''


def relu(x):
    return np.maximum(x, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def drelu(grad_a, act):
    grad_a[act <= 0] = 0
    return grad_a

def dsigmoid(grad_a, act):
    return np.multiply(grad_a, act - np.square(act))

def dtanh(grad_a, act):
    return np.multiply(grad_a, 1 - np.square(act))


class MLP(object):
    def __init__(self, act_type, opt_type, layers):
        act_funcs = {'ReLU': relu, 'Sigmoid': sigmoid, 'Tanh': tanh}
        dacts = {'ReLU': drelu, 'Sigmoid': dsigmoid, 'Tanh': dtanh}
        optimizers = {'SGD': self.sgd, 'Momentum': self.momentum, 'Nesterov': self.nesterov, 
                      'AdaGrad': self.adagrad, 'RMSprop': self.rmsprop, 'Adam': self.adam}

        self.reg = 2 #0=none, 1=L1, 2=L2 regularization
        self.lmbda = 1e-4 # regularization coefficient
        self.gamma = 0.9
        self.eps = 1e-8
        self.epochs, self.batch_size = 200, 16
        self.learning_rate = 0.2
        self.layer_num = len(layers)-1

        self.afunc = act_funcs[act_type]
        self.dact = dacts[act_type]
        self.opt = optimizers[opt_type]

        # Randomly initialize weights
        self.w, self.b = [np.empty]*self.layer_num, [np.empty]*self.layer_num
        self.mom_w, self.cache_w = [np.empty]*self.layer_num, [np.empty]*self.layer_num
        self.mom_b, self.cache_b = [np.empty]*self.layer_num, [np.empty]*self.layer_num

        for i in range(self.layer_num):
            self.w[i] = np.random.randn(layers[i], layers[i+1])
            self.b[i] = np.random.randn(1, layers[i+1])
            self.mom_w[i] = np.zeros_like(self.w[i])
            self.cache_w[i] = np.zeros_like(self.w[i])
            self.mom_b[i] = np.zeros_like(self.b[i])
            self.cache_b[i] = np.zeros_like(self.b[i])


    def sgd(self, grad_w, grad_b):
        alpha = self.learning_rate / self.batch_size
        for i in range(self.layer_num):
            self.w[i] -= alpha * grad_w[i]
            self.b[i] -= alpha * grad_b[i]

    def momentum(self, grad_w, grad_b):
        alpha = self.learning_rate / self.batch_size
        for i in range(self.layer_num):
            self.mom_w[i] = self.gamma * self.mom_w[i] + alpha * grad_w[i]
            self.w[i] -= self.mom_w[i]
            self.mom_b[i] = self.gamma * self.mom_b[i] + alpha * grad_b[i]
            self.b[i] -= self.mom_b[i]

    def nesterov(self, grad_w, grad_b):
        alpha = self.learning_rate / self.batch_size
        for i in range(self.layer_num):
            mom_v_prev = self.mom_w[i]
            self.mom_w[i] = self.gamma * self.mom_w[i] + alpha * grad_w[i]
            self.w[i] -= ((1 + self.gamma) * self.mom_w[i] - self.gamma * mom_v_prev)
            mom_b_prev = self.mom_b[i]
            self.mom_b[i] = self.gamma * self.mom_b[i] + alpha * grad_b[i]
            self.b[i] -= ((1 + self.gamma) * self.mom_b[i] - self.gamma * mom_b_prev)

    def adagrad(self, grad_w, grad_b):
        alpha = self.learning_rate / self.batch_size
        for i in range(self.layer_num):
            self.cache_w[i] += np.square(grad_w[i])
            self.w[i] -= alpha * grad_w[i] / (np.sqrt(self.cache_w[i]) + self.eps)
            self.cache_b[i] += np.square(grad_b[i])
            self.b[i] -= alpha * grad_b[i] / (np.sqrt(self.cache_b[i]) + self.eps)

    def rmsprop(self, grad_w, grad_b):
        alpha = self.learning_rate / self.batch_size
        for i in range(self.layer_num):
            self.cache_w[i] = self.gamma * self.cache_w[i] + (1-self.gamma) * np.square(grad_w[i])
            self.w[i] -= alpha * grad_w[i] / (np.sqrt(self.cache_w[i]) + self.eps)
            self.cache_b[i] = self.gamma * self.cache_b[i] + (1-self.gamma) * np.square(grad_b[i])
            self.b[i] -= alpha * grad_b[i] / (np.sqrt(self.cache_b[i]) + self.eps)

    def adam(self, grad_w, grad_b):
        beta1 = 0.9
        beta2 = 0.999
        alpha = self.learning_rate / self.batch_size
        for i in range(self.layer_num):
            self.mom_w[i] = beta1 * self.mom_w[i] + (1 - beta1) * grad_w[i]
            self.cache_w[i] = beta2 * self.cache_w[i] + (1 - beta2) * np.square(grad_w[i])
            self.w[i] -= alpha * self.mom_w[i] / (np.sqrt(self.cache_w[i]) + self.eps)
            self.mom_b[i] = beta1 * self.mom_b[i] + (1 - beta1) * grad_b[i]
            self.cache_b[i] = beta2 * self.cache_b[i] + (1 - beta2) * np.square(grad_b[i])
            self.b[i] -= alpha * self.mom_b[i] / (np.sqrt(self.cache_b[i]) + self.eps)

    def regularization(self):
        if(self.reg == 0):
            return
        alpha = self.learning_rate * self.lmbda
        for i in range(self.layer_num):
            if(self.reg==1):
                self.w[i] -= alpha * np.sign(self.w[i])
            elif(self.reg==2):
                self.w[i] -= alpha * self.w[i]

    def predict(self, x):
        act = x
        for i in range(self.layer_num-1):
            act = self.afunc(act.dot(self.w[i]) + self.b[i])
        return act.dot(self.w[self.layer_num-1]) + self.b[self.layer_num-1]

    def fit(self, x, y):
        train_num = x.shape[0]
        l_num = self.layer_num
        bvec = np.ones((1, self.batch_size))
        for epoch in range(self.epochs):
            #mini batch
            permut=np.random.permutation(train_num//self.batch_size*self.batch_size).reshape(-1,self.batch_size)
            for b_idx in range(permut.shape[0]):
                # Forward pass: compute predicted out
                act = [np.empty]*(l_num+1)
                act[0] = x[permut[b_idx,:]]
                for i in range(1, l_num):
                    act[i] = self.afunc(act[i-1].dot(self.w[i-1]) + self.b[i-1])
                act[l_num] = act[l_num-1].dot(self.w[l_num-1]) + self.b[l_num-1]

                # Backprop to compute gradients of weights & activaions
                grad_a, grad_w, grad_b = [np.empty]*(l_num+1), [np.empty]*l_num, [np.empty]*l_num
                grad_a[l_num] = act[l_num] - y[permut[b_idx,:]]
                grad_w[l_num-1] = act[l_num-1].T.dot(grad_a[l_num])
                grad_b[l_num-1] = bvec.dot(grad_a[l_num])

                for i in reversed(range(1, l_num)):
                    grad_a[i] = grad_a[i+1].dot(self.w[i].T)
                    grad_a[i] = self.dact(grad_a[i], act[i])
                    grad_w[i-1] = act[i-1].T.dot(grad_a[i])
                    grad_b[i-1] = bvec.dot(grad_a[i])

                # Update weights
                self.regularization()
                self.opt(grad_w, grad_b)

            # Compute loss and visualization
            if (epoch+1)%50==0:
                evol_boundary_vis(self, x, y, epoch, [2, 2, (epoch+1)//50])


def main():
    x, y = gen_spiral_data(512)
    mlp = MLP('Tanh', 'Adam', layers=[2,8,7,1])
    mlp.fit(x, y)
    
if __name__ == "__main__":
    main()