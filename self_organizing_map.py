import numpy as np
import matplotlib.pyplot as plt


class SOM(object):

    def __init__(self):
        self.sigma = 1
        self.lr = 0.1
        self.eps = 0.05
        self.n_size = 10
        self.iterations = 10
        self.neighbors_radius = []
        radius = 4
        for i in range(-radius, radius+1):
            for j in range(-radius, radius+1):
                if i * i + j * j <= radius * radius:
                    self.neighbors_radius.append((i, j))
        self.w = None

    def get_bmu(self, w, x):
        dist = np.square(w - x).sum(axis=2)
        index = np.argmin(dist)
        return np.array([index // self.n_size, index % self.n_size])

    def fit(self, x):
        fig, ax = plt.subplots(nrows=2, ncols=5, subplot_kw=dict(xticks=[], yticks=[]))

        self.w = np.random.randn(self.n_size, self.n_size, x.shape[1])
        sigma_sq = self.sigma * self.sigma
        for step in range(self.iterations):
            for y in np.random.permutation(x):
                i, j = self.get_bmu(self.w, y)
                # update w
                for di, dj in self.neighbors_radius:
                    if i + di >= 0 and i + di < self.n_size and j + di >= 0 and j + dj < self.n_size: 
                        self.w[i + di][j + dj] += self.lr * (y - self.w[i + di][j + dj]) * np.exp(-np.square([di, dj]).sum() / 2 / sigma_sq)
            self.lr *= np.exp(-step * self.eps)
            sigma_sq *= np.exp(-step * self.eps)
            ax[step//5][step%5].imshow(self.w.astype(int))
            ax[step//5][step%5].title.set_text(step)
        plt.show()
        return self.w

def main():
    som = SOM()
    x = np.random.randint(0, 255, (3000, 3))
    w = som.fit(x)


if __name__ == "__main__":
    main()
