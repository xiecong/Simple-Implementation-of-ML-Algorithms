import numpy as np
# todo: update examples


class HMM(object):

    def __init__(self, o_num, s_num):
        self.o_num = o_num
        self.s_num = s_num
        self.A = np.random.rand(s_num, s_num)  # np.ones((s_num, s_num))/s_num#
        self.A = self.A / self.A.sum(axis=1).reshape(-1, 1)
        self.B = np.random.rand(s_num, o_num)  # np.ones((s_num, o_num))/o_num#
        self.B = self.B / self.B.sum(axis=1).reshape(-1, 1)
        self.pi = np.ones(s_num) / s_num

    def fit(self, obs):
        self.baum_welch(obs)

    def predict(self, obs):
        return self.viterbi(obs)

    # Probability of an observed sequence
    def forward(self, obs):
        alpha = np.zeros((obs.shape[0], self.s_num))
        alpha[0, :] = np.multiply(self.pi, self.B[:, obs[0]])
        for i in range(1, len(obs)):
            alpha[i, :] = np.multiply(
                np.matmul(alpha[i - 1, :], self.A), self.B[:, obs[i]])
        return alpha

    def backward(self, obs):
        beta = np.zeros((obs.shape[0] + 1, self.s_num))
        beta[obs.shape[0], :] = np.ones((1, self.s_num))
        for i in range(len(obs) - 1, -1, -1):
            beta[i, :] = np.matmul(
                self.A, beta[i + 1, :].T * self.B[:, obs[i]])
        return beta

    def baum_welch(self, obs, epsilon=0.05, max_it=100):
        it = 0
        obs_indicator = np.zeros((len(obs), self.o_num))
        obs_indicator[np.arange(len(obs)), obs] = 1
        error = epsilon + 1
        while(error > epsilon and it < 100):
            alpha = self.forward(obs)
            beta = self.backward(obs)[1:]
            # E step
            xi = np.zeros((self.s_num, self.s_num))
            likelihood = (alpha * beta).T
            gamma = likelihood / likelihood.sum(axis=0).reshape(1, -1)
            xi = alpha[:-1].T.dot((beta[1:] / likelihood[:, :-1].sum(
                axis=0).reshape(-1, 1)) * self.B[:, obs[1:]].T) * self.A

            # M step
            self.pi = (beta[1, :] * self.B[:, obs[1]] / likelihood[:,
                                                                   0].sum()).dot((self.A * alpha[0, :].reshape(-1, 1)).T)
            A = xi / xi.sum(axis=1).reshape(-1, 1)
            B = gamma.dot(obs_indicator) / gamma.sum(axis=1).reshape(-1, 1)

            error = (np.abs(A - self.A)).max() + (np.abs(B - self.B)).max()
            it += 1
            self.A, self.B = A, B

    def viterbi(self, obs):
        v = np.multiply(self.pi, self.B[:, obs[0]])
        vpath = np.arange(self.s_num).reshape(-1, 1).tolist()
        for i in range(1, len(obs)):
            prev = np.array([np.argmax(v * self.A[:, n])
                             for n in range(self.s_num)])
            v = v[prev] * self.A.flatten()[prev * self.s_num +
                                           np.arange(self.s_num)] * self.B[:, obs[i]]
            vpath = [vpath[prev[i]] + [i] for i in range(self.s_num)]
        return vpath[np.argmax(v)]


def seq_generator():
    o_num, s_num = 2, 2
    A = np.array([[0.4, 0.6], [0.9, 0.1]])
    B = np.array([[0.49, 0.51], [0.85, 0.15]])
    pi = np.array([0.5, 0.5])
    #obs = np.array([0,1,1,0,1,1,0,0,1,1,0,1,1,1,0,0,1,0,0,1,1,0,1,1,1,1,0,1,0,0,1,0,1,0,0,1,1,1,0])
    '''
	hmm.pi = np.array([0.25, 0.25, 0.25, 0.25])
	hmm.A = np.array([[0.05,0.7, 0.05, 0.2],[0.1,0.4,0.3,0.2],[0.1,0.6,0.05,0.25],[0.25,0.3,0.4,0.05]])
	hmm.B = np.array([[0.3,0.4,0.2,0.1],[0.2,0.1,0.2,0.5],[0.4,0.2,0.1,0.3],[0.3,0.05,0.3,0.35]])
	obs = np.array([3,1,1,3,0,3,3,3,1,1,0,2,2])
	'''
    '''
	A = np.array([[.4,.3,.1,.2],[.6,.05,.1,.25],[.7,.05,.05,.2],[.3,.4,.25,.05]])
	B = np.array([[.2,.1,.2,.5],[.4,.2,.1,.3],[.3,.4,.2,.1],[.3,.05,.3,.35]])
	pi = np.array([0.25,0.25,0.25,0.25])
	'''
    q = np.random.choice(s_num, 1, p=pi)[0]
    v = []
    for i in range(100):
        v.append(np.random.choice(o_num, 1, p=B[q].flatten())[0])
        q = np.random.choice(s_num, 1, p=A[q].flatten())[0]
    # print(np.array(q))
    obs = np.array(v)
    return obs, A, B, pi


def main():
    hmm1 = HMM(2, 2)
    obs, hmm1.A, hmm1.B, hmm1.pi = seq_generator()
    print(hmm1.forward(obs))
    print(hmm1.predict(obs))

    hmm2 = HMM(2, 2)
    hmm2.baum_welch(obs)
    print(hmm2.pi)
    print(hmm2.A)
    print(hmm2.B)


if __name__ == "__main__":
    main()
