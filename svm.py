import numpy as np
from sklearn.datasets import load_breast_cancer
from decision_boundary_vis import gen_xor_data, gen_spiral_data, gen_circle_data, gen_linear_data, boundary_vis


class SVM(object):
	def __init__(self):
		self.b = 0
		self.kernel = self.gaussian
		self.sigma = 1
		self.p = 2
		self.C = 1

	def _select_pair(self, u, y, alpha):
		n_data = len(y)
		uy = u * y

		violations = np.zeros(len(y))
		violations[uy>=1] = self.C - alpha[uy>=1]
		violations[uy<=1] = alpha[uy<=1]
		violations[uy==1] = ((alpha[uy==1] == self.C) + (alpha[uy==1] == 0)) * self.C / 2
		if violations.max() == 0:
			return -1, -1
		idx1 = np.random.choice(n_data, 1, p=violations/violations.sum()).sum()
		diff = abs(u - y - u[idx1] + y[idx1])
		idx2 = np.random.choice(n_data, 1, p=diff/diff.sum()).sum()
		print(idx1, idx2, alpha[idx1], alpha[idx2], violations[idx1])
		return idx1, idx2

	def fit(self, x, y):  # SMO
		n_data = x.shape[0]
		self.supp_w = np.ones(x.shape[0])
		self.supp_x = x
		self.b = 0
		alpha = np.zeros(n_data)
		for i in range(200):
			# select alpha1, alpha2
			u = self.predict(x)
			idx1, idx2 = self._select_pair(u, y, alpha)
			if(idx1 == -1): break
			y1, y2 = y[idx1], y[idx2]

			# update alpha1, alpha2
			if y1 != y2:
				L = max(0.0, alpha[idx2] - alpha[idx1])
				H = min(self.C, self.C + alpha[idx2] - alpha[idx1])
			else:
				L = max(0.0, alpha[idx1] + alpha[idx2] - self.C)
				H = min(self.C, alpha[idx1] + alpha[idx2])
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
			sv = np.ones(n_data).astype(bool) if alpha.sum() == 0 else (alpha!=0)
			self.supp_w = alpha[sv] * y[sv] #alpha[alpha!=0] * y[alpha!=0]
			self.supp_x = x[sv] # x[alpha!=0]
		print(x[sv])

	def predict(self, x):
		return self.supp_w.dot(self.kernel(self.supp_x, x)).flatten() + self.b

	def gaussian(self, x1, x2):
		sub = np.array([[np.square(x1i-x2i).sum() for x2i in x2] for x1i in x1])
		sigma_sq=self.sigma**2
		return np.exp(-sub / 2 / sigma_sq)

	def polynomial(self, x1, x2):
		return (x1.dot(x2.T) + 1)**self.p

	def linear(self, x1, x2):
		return x1.dot(x2.T)


def main():
	data, target = gen_circle_data(200) # load_breast_cancer() #
	#target = data.target * 2 - 1
	test_ratio = 0.2  # y set to -1
	test_split = np.random.uniform(0, 1, len(target))
	train_x = data[test_split >= test_ratio]
	test_x = data[test_split < test_ratio]
	train_y = target[test_split >= test_ratio].flatten()
	test_y = target[test_split < test_ratio]

	svm = SVM()
	svm.fit(train_x, train_y)
	print(sum(np.sign(svm.predict(train_x)) == train_y)/train_x.shape[0])
	print(sum(np.sign(svm.predict(test_x)) == test_y.flatten())/test_x.shape[0])
	boundary_vis(svm, test_x, test_y)

if __name__ == "__main__":
	main()