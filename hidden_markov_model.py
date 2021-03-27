import numpy as np
# todo: update examples


class HMM(object):

    def __init__(self, o_num, s_num, pi=None, A=None, B=None):
        self.o_num = o_num
        self.s_num = s_num
        self.A = np.random.rand(s_num, s_num) if A is None else A
        self.A = self.A / self.A.sum(axis=1).reshape(-1, 1)
        self.B = np.random.rand(s_num, o_num) if B is None else B
        self.B = self.B / self.B.sum(axis=1).reshape(-1, 1)
        self.pi = np.ones(s_num) / s_num if pi is None else pi

    # Probability of an observed sequence
    def forward(self, obs):
        alpha = np.zeros((obs.shape[0], self.s_num))
        alpha[0, :] = self.pi * self.B[:, obs[0]]
        for t in range(1, len(obs)):
            alpha[t, :] = alpha[t - 1, :].dot(self.A) * self.B[:, obs[t]]
        return alpha

    def backward(self, obs):
        beta = np.zeros((obs.shape[0], self.s_num))
        beta[obs.shape[0] - 1, :] = np.ones((1, self.s_num))
        for t in range(obs.shape[0] - 2, -1, -1):
            beta[t, :] = self.A.dot((beta[t + 1, :] * self.B[:, obs[t + 1]]).T)
        return beta

    def baum_welch(self, obs, epsilon=0.05, max_it=100):
        it = 0
        obs_indicator = np.zeros((len(obs), self.o_num))
        obs_indicator[np.arange(len(obs)), obs] = 1
        error = epsilon + 1
        while(error > epsilon and it < 100):
            alpha = self.forward(obs)
            beta = self.backward(obs)

            # E step
            xi = np.zeros((self.s_num, self.s_num))
            likelihood = (alpha * beta).T
            gamma = likelihood / likelihood.sum(axis=0).reshape(1, -1)
            for t in range(0, len(obs) - 1):
                xit = alpha[
                    t].reshape(-1, 1).dot((beta[t + 1] * self.B[:, obs[t + 1]]).reshape(1, -1)) * self.A
                xi += xit / xit.sum()

            # M step
            self.pi = gamma[:, 0]
            A = xi / gamma[:, :-1].sum(axis=1).reshape(-1, 1)
            B = gamma.dot(obs_indicator) / gamma.sum(axis=1).reshape(-1, 1)

            error = (np.abs(A - self.A)).max() + (np.abs(B - self.B)).max()
            it += 1
            self.A, self.B = A, B

    def viterbi(self, obs):
        v = self.pi * self.B[:, obs[0]]
        vpath = np.arange(self.s_num).reshape(-1, 1).tolist()
        for i in range(1, len(obs)):
            prev = np.array([np.argmax(v * self.A[:, n])
                             for n in range(self.s_num)])
            v = v[prev] * self.A[prev,
                                 np.arange(self.s_num)] * self.B[:, obs[i]]
            vpath = [vpath[prev[s]] + [s] for s in range(self.s_num)]
        return vpath[np.argmax(v)]


def seq_generator():
    o_num, s_num = 2, 2
    A = np.array([[0.4, 0.6], [0.9, 0.1]])
    B = np.array([[0.49, 0.51], [0.85, 0.15]])
    pi = np.array([0.5, 0.5])
    q = np.random.choice(s_num, 1, p=pi)[0]
    v = []
    for i in range(100):
        v.append(np.random.choice(o_num, 1, p=B[q].flatten())[0])
        q = np.random.choice(s_num, 1, p=A[q].flatten())[0]
    obs = np.array(v)
    return obs, A, B, pi


def main():
    hmm = HMM(
        o_num=3, s_num=2,
        pi=np.array([0.6, 0.4]),
        A=np.array([[0.7, 0.3], [0.4, 0.6]]),
        B=np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])
    )
    # 0, 0, 1 see example in https://en.wikipedia.org/wiki/Viterbi_algorithm
    print('viterbi', hmm.viterbi([2, 1, 0]))

    # examples here https://iulg.sitehost.iu.edu/moss/hmmcalculations.pdf
    hmm = HMM(
        o_num=2, s_num=2,
        pi=np.array([0.85, 0.16]),
        A=np.array([[0.3, 0.7], [0.1, 0.9]]),
        B=np.array([[0.4, 0.6], [0.5, 0.5]])
    )
    obs = np.array([0, 1, 1, 0])
    hmm.baum_welch(obs)
    print('initial probabilities', hmm.pi)
    print('transition matrix', hmm.A)
    print('emission matrix', hmm.B)


if __name__ == "__main__":
    main()
