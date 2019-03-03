import numpy as np
from sklearn import datasets
class DecisionStump(object):
	def __init__(self, x, y, w):
		self.feature = None
		self.value = None
		self.l_value = None
		self.r_value = None
		self.x = x
		self.y = y
		self.w = w

	def fit(self):
		min_err = np.float("inf")
		for f in range(self.x.shape[1]):
			split_values = np.unique(self.x[:,f].round(decimals=4))
			for split_value in split_values:
				l_value, r_value, err = self.split(f, split_value)
				if err < min_err:
					min_err = err
					self.l_value, self.r_value = l_value, r_value
					self.feature, self.value = f, split_value
		#print(self.feature, self.value, self.l_value, self.r_value)

	def split(self, feature, value):
		f_vec = self.x[:, feature]
		l_value = np.sign(self.y[np.nonzero(f_vec<value)].sum())
		r_value = np.sign(self.y[np.nonzero(f_vec>=value)].sum())
		pred = (f_vec<value)*l_value + (f_vec>=value)*r_value
		error = (pred!=self.y).dot(self.w.T)
		return l_value, r_value, error

	def predict(self, x):
		f_vec = x[:, self.feature]
		l_idxes = np.nonzero(f_vec<self.value)
		r_idxes = np.nonzero(f_vec>=self.value)
		return (f_vec<self.value)*self.l_value + (f_vec>=self.value)*self.r_value

class Adaboost(object):
	def __init__(self, x, y):
		self.esti_num = 10
		self.estimators = []
		self.alphas = []
		self.x = x
		self.y = y
		self.w = np.ones(self.x.shape[0])/self.x.shape[0]

	def fit(self):
		eps = 1e-16
		prediction = np.zeros(self.x.shape[0])
		for i in range(self.esti_num):
			self.estimators.append(DecisionStump(self.x, self.y, self.w))
			self.estimators[i].fit()
			pred_i = self.estimators[i].predict(self.x)
			error_i = (pred_i!=self.y).dot(self.w.T)
			self.alphas.append(np.log((1.0-error_i)/(error_i+eps))/2)
			self.w = np.multiply(self.w, np.exp(self.alphas[i]*2*(pred_i!=self.y)-1))
			self.w = self.w / self.w.sum()

			prediction += pred_i * self.alphas[i]
			print("Tree {} constructed, acc {}".format(i, (np.sign(prediction)==self.y).sum()/self.x.shape[0]))

	def predict(self, x):
		return np.sign(sum(esti.predict(x) * alpha for esti, alpha in zip(self.estimators, self.alphas)))

def main():
	data = datasets.load_breast_cancer()
	x = data.data
	y = data.target*2-1
	adaboost = Adaboost(x, y)
	adaboost.fit()
	print((adaboost.predict(x)==y).sum()/x.shape[0])

if __name__ == "__main__":
    main()