import numpy as np
from sklearn.datasets import load_breast_cancer

    
class SVM(object):
	def __init__(self):
		self.b = 0
		self.kernel = self.polynomial
		self.gamma = 1
		self.degree = 3
		self.C = 1
    
	def _ktt_violations(self, uy, alpha):
		violations = np.zeros(len(uy))
		violations[uy>=1] = self.C - alpha[uy>=1]
		violations[uy<=1] = alpha[uy<=1]
		violations[uy==1] = ((alpha[uy==1] >= self.C) + (alpha[uy==1] <= 0)) * self.C / 2
		return violations
        
	def _select_pair_by_delta_e(self, u, y, alpha):
		violations = self._ktt_violations(u * y, alpha) > 0
		if violations.max() == 0: return -1, -1
		e = u - y
		repeat_e = np.repeat(e.reshape(1, -1), e.shape[0], axis=0)
		delta_e = (violations * abs((repeat_e - repeat_e.T))).flatten()
		idx = np.random.choice(len(delta_e), 1, p=delta_e / delta_e.sum()).sum()
		return idx % len(e), idx // len(e)

	def _select_pair_by_max_violations(self, u, y, alpha):
		n_data = len(y)
		violations = self._ktt_violations(u * y, alpha)
		if violations.max() == 0: return -1, -1
		idx1 = np.random.choice(n_data, 1, p=violations / violations.sum()).sum()
		delta_e = abs(u - y - u[idx1] + y[idx1])
		idx2 = np.random.choice(n_data, 1, p=delta_e / delta_e.sum()).sum()
		return idx1, idx2
        
	def loss(self, alpha, x, y):
		w = np.matmul(self.supp_w.reshape(-1, 1), self.supp_w.reshape(1, -1))
		return alpha.sum() - (w * self.kernel(self.supp_x, self.supp_x)).sum() / 2

	def fit(self, x, y):  # SMO
		n_data = x.shape[0]
		self.supp_w = np.ones(x.shape[0])
		self.supp_x = x
		self.b = 0
		alpha = np.zeros(n_data)
		for i in range(500):
			# select alpha1, alpha2
			u = np.sign(self.predict(x))
			idx1, idx2 = self._select_pair_by_max_violations(u, y, alpha)
			if(idx1 == -1): break
			y1, y2 = y[idx1], y[idx2]

			# update alpha1, alpha2
			L = max(0, alpha[idx2] - alpha[idx1]) if y1 != y2 else max(0, alpha[idx1] + alpha[idx2] - self.C)
			H = min(self.C, self.C + alpha[idx2] - alpha[idx1]) if y1 != y2 else min(self.C, alpha[idx1] + alpha[idx2])
			e1, e2 = u[idx1] - y1, u[idx2] - y2
			k11 = self.kernel(x[[idx1]], x[[idx1]]).sum()
			k12 = self.kernel(x[[idx1]], x[[idx2]]).sum()
			k22 = self.kernel(x[[idx2]], x[[idx2]]).sum()
			alpha2 = min(H, max(L, alpha[idx2] + y2 * (e1 - e2) / (k11 + k22 - 2 * k12)))
			alpha1 = alpha[idx1] + y1 * y2 * (alpha[idx2] - alpha2)

			# update b
			b1 = self.b - e1 - y1 * (alpha1 - alpha[idx1]) * k11 - y2 * (alpha2 - alpha[idx2]) * k12
			b2 = self.b - e2 - y1 * (alpha1 - alpha[idx1]) * k12 - y2 * (alpha2 - alpha[idx2]) * k22
			if alpha1 > 0 and alpha1 < self.C: self.b = b1
			elif alpha2 > 0 and alpha2 < self.C: self.b = b2
			else: self.b = (b1 + b2) / 2

			# update model
			alpha[[idx1, idx2]] = [alpha1, alpha2]
			sv = np.array([True]) if alpha.sum() == 0 else (alpha!=0)
			self.supp_w = alpha[sv] * y[sv]
			self.supp_x = x[sv]
			if i % 100 == 0:
				print(self.loss(alpha, x, y))
		print('support vectors:', x[sv])

	def predict(self, x):
		return self.supp_w.dot(self.kernel(self.supp_x, x)).flatten() + self.b

	def rbf(self, x1, x2):
		sub = np.array([[np.square(x1i-x2i).sum() for x2i in x2] for x1i in x1])
		return np.exp(-self.gamma * sub)

	def polynomial(self, x1, x2):
		return (x1.dot(x2.T) + 1)**self.degree

	def linear(self, x1, x2):
		return x1.dot(x2.T)


def main():
	data = load_breast_cancer()
	target = data.target * 2 - 1
	test_ratio = 0.2
	test_split = np.random.uniform(0, 1, len(target))
	train_x = data.data[test_split >= test_ratio]
	test_x = data.data[test_split < test_ratio]
	train_y = target[test_split >= test_ratio]
	test_y = target[test_split < test_ratio]
	svm = SVM()
	svm.fit(train_x, train_y)
	print(sum(np.sign(svm.predict(train_x)) == train_y)/train_x.shape[0])
	print(sum(np.sign(svm.predict(test_x)) == test_y)/test_x.shape[0])


if __name__ == "__main__":
	main()
