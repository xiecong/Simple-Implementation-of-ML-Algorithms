import numpy as np
import matplotlib.pyplot as plt
# implements and visualize Gradient Descent, Momentum, Nesterov, AdaGrad,
# RMSprop, Adam, and Simulated Annealing


class Optimization(object):

    def __init__(self):
        self.optimizers = {'GD': self.gradient_descent, 'Momentum': self.momentum, 'Nesterov': self.nesterov,
                           'AdaGrad': self.adagrad, 'RMSprop': self.rmsprop, 'Adam': self.adam}
        self.gamma = 0.8
        self.eps = 1e-8
        self.reset()

    def reset(self, lr=0.001):
        self.pos = np.array([-10.0, -5.0])
        self.mom = np.zeros_like(self.pos)
        self.cache = np.zeros_like(self.pos)
        self.adam_iter = 1
        self.learning_rate = lr

    def gradient_descent(self, grad):
        self.pos -= self.learning_rate * grad

    def momentum(self, grad):
        self.mom = self.gamma * self.mom + self.learning_rate * grad
        self.pos -= self.mom

    def nesterov(self, grad):
        mom_v_prev = self.mom
        self.mom = self.gamma * self.mom + self.learning_rate * grad
        self.pos -= ((1 + self.gamma) * self.mom - self.gamma * mom_v_prev)

    def adagrad(self, grad):
        self.cache += np.square(grad)
        self.pos -= self.learning_rate * grad / \
            (np.sqrt(self.cache) + self.eps)

    def rmsprop(self, grad):
        self.cache = self.gamma * self.cache + \
            (1 - self.gamma) * np.square(grad)
        self.pos -= self.learning_rate * grad / \
            (np.sqrt(self.cache) + self.eps)

    def adam(self, grad):
        beta1 = 0.5
        beta2 = 0.8
        self.mom = beta1 * self.mom + (1 - beta1) * grad
        self.cache = beta2 * self.cache + (1 - beta2) * np.square(grad)
        self.pos -= self.learning_rate * self.mom / \
            (1 - beta1**self.adam_iter) / \
            (np.sqrt(self.cache / (1 - beta2**self.adam_iter)) + self.eps)
        self.adam_iter += 1

    def optimize(self, opt_algo, grad_func, x, y):
        trace = [self.pos.copy()]
        for i in range(30):
            grad = grad_func(self.pos, x, y)
            self.optimizers[opt_algo](grad)
            if np.sum(np.square(self.pos - np.array([3, 5]))) < 1:
                break
            trace.append(self.pos.copy())
        return np.array(trace)


class Annealing(object):

    def __init__(self):
        self.learning_rate = 0.5
        self.pos = np.array([-10.0, -5.0])
        self.iterations = 100

    def transfer_prob(self, e_old, e_new, t):
        if e_old > e_new:
            return 1
        else:
            return np.exp((e_old - e_new) / t)

    def annealing(self, x, y):
        trace = [self.pos.copy()]
        for i in range(self.iterations):
            t = 1 - i / self.iterations
            radius, theta = 5 * self.learning_rate * np.random.uniform(), np.random.uniform() * \
                2 * np.pi - np.pi
            pos_next = self.pos + radius * \
                np.array([np.cos(theta), np.sin(theta)])
            p = self.transfer_prob(loss(self.pos, x, y),
                                   loss(pos_next, x, y), t)
            if p >= np.random.uniform():
                self.pos = pos_next
                trace.append(self.pos.copy())
            if np.sum(np.square(self.pos - np.array([3, 5]))) < 1:
                break
        return np.array(trace)


def loss(w, x, y):  # w:1*2
    return np.mean(np.square(w.reshape((1, 2)).dot(x) - y)) / 2


def grad(w, x, y):
    y_hat = w.dot(x)
    return (y_hat - y).reshape(1, -1).dot(x.T).flatten()


def main():
    dim = 400
    x = np.linspace(-1, 1, dim)
    y = 3 * x + 5 + np.random.randn(dim)
    x_expand = np.concatenate([x.reshape((1, dim)), np.ones((1, dim))], axis=0)
    w_mesh, b_mesh = np.meshgrid(
        np.linspace(-12, 15, 100), np.linspace(-5, 15, 100))
    loss_grid = np.array([
        loss(np.array([w, b]), x_expand, y)
        for w, b in zip(np.ravel(w_mesh), np.ravel(b_mesh))
    ])
    plt.contour(w_mesh, b_mesh, loss_grid.reshape(
        w_mesh.shape), 70, cmap='bwr_r', alpha=0.5)
    opt = Optimization()
    an = Annealing()
    for i, opt_algo, lr in zip(range(7), ['GD', 'Momentum', 'Nesterov', 'AdaGrad', 'RMSprop', 'Adam', 'Annealing'], [0.0035, 0.0005, 0.0006, 10, 2, 5, 0.5]):
        if opt_algo == 'Annealing':
            trace = an.annealing(x_expand, y)
        else:
            opt.reset(lr)
            trace = opt.optimize(opt_algo, grad, x_expand, y)
        print(f'{opt_algo} finished with {trace.shape[0]} steps')
        angles = trace[1:] - trace[:-1]
        q = plt.quiver(trace[:-1, 0], trace[:-1, 1], angles[:, 0], angles[:, 1],
                       scale_units='xy', angles='xy', scale=1, color=plt.cm.get_cmap('Set1')(i), alpha=1, width=0.004)
        plt.quiverkey(q, X=1.06, Y=0.9 - i * 0.1, U=1, label=opt_algo)
    plt.show()


if __name__ == "__main__":
    main()
