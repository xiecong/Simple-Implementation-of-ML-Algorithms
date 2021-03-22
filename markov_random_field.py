import numpy as np
from skimage import data
import matplotlib.pyplot as plt
from skimage.transform import resize
# Markov Random Field for image segmentation


class MRF(object):

    def __init__(self, img):
        self.max_iter = 100000
        self.beta = 100
        self.n_label = 3
        self.n_col, self.n_row = img.shape[1], img.shape[0]
        # init label by value
        self.labels = np.ones(self.n_col * self.n_row).astype(int)
        self.labels[img.flatten() <= np.quantile(img, 0.33)] = 0
        self.labels[img.flatten() >= np.quantile(img, 0.67)] = 2
        self.label_means, self.label_vars = self.get_label_stats(
            img.flatten(), self.labels)

    def energy(self, img):
        energy = 0
        for idx in range(self.n_row * self.n_col):
            # sum p(x_s, y_s)
            mean, var = self.label_means[self.labels[
                idx]], self.label_vars[self.labels[idx]]
            energy += np.log(np.sqrt(2 * np.pi * var)) + \
                np.square(img[idx] - mean) / 2 / var
            # sum p(x_s, x_t) t in s neighbors
            for di, dj in [[0, -1], [-1, 0], [0, 1], [1, 0]]:
                if not(0 <= idx // self.n_col + di < self.n_row and 0 <= idx % self.n_col + dj < self.n_col):
                    continue
                energy += -self.beta / 2 if self.labels[idx] == self.labels[
                    idx + di * self.n_col + dj] else self.beta / 2
        return energy

    def get_label_stats(self, img, labels):
        return [np.mean(img[labels == i]) for i in range(self.n_label)], [np.var(img[labels == i]) for i in range(self.n_label)]

    def transition_prob(self, img, idx, new_label, t):
        new_labels = self.labels.copy()
        new_labels[idx] = new_label
        old_mean, old_var = self.label_means[
            self.labels[idx]], self.label_vars[self.labels[idx]]
        new_mean, new_var = np.mean(img[new_labels == new_label]), np.var(
            img[new_labels == new_label])
        delta_energy = np.log(np.sqrt(2 * np.pi * new_var)) + \
            np.square(img[idx] - new_mean) / 2 / new_var
        delta_energy -= np.log(np.sqrt(2 * np.pi * old_var)) + \
            np.square(img[idx] - old_mean) / 2 / old_var

        for di, dj in [[0, -1], [-1, 0], [0, 1], [1, 0], [-1, -1], [-1, 1], [1, 1], [1, -1]]:
            if not(0 <= idx // self.n_col + di < self.n_row and 0 <= idx % self.n_col + dj < self.n_col):
                continue
            delta_energy += -self.beta if new_label == self.labels[
                idx + di * self.n_col + dj] else self.beta
            delta_energy -= -self.beta if self.labels[idx] == self.labels[
                idx + di * self.n_col + dj] else self.beta
        if delta_energy < 0:
            return 1
        else:
            return np.exp(-delta_energy / t)

    def optimize(self, img):
        for t in range(self.max_iter):
            idx = np.random.choice(img.shape[0])
            lp = np.ones(self.n_label)
            lp[self.labels[idx]] = 0
            new_label = np.random.choice(self.n_label, p=lp / lp.sum())
            prob = self.transition_prob(
                img=img, idx=idx, new_label=new_label, t=0.01 * (1 - t / self.max_iter))
            if prob >= np.random.uniform():
                self.labels[idx] = new_label
                self.label_means, self.label_vars = self.get_label_stats(
                    img, self.labels)
        return self.labels


def main():
    img = data.camera()
    img = resize(
        img, (img.shape[0] // 5, img.shape[1] // 5), anti_aliasing=True)
    mrf = MRF(img)
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.imshow(mrf.labels.reshape((img.shape[0], -1)), cmap='gray')
    seg_img = mrf.optimize(img.flatten())
    plt.subplot(1, 3, 3)
    plt.imshow(seg_img.reshape((img.shape[0], -1)), cmap='gray')
    plt.show()


if __name__ == "__main__":
    main()
