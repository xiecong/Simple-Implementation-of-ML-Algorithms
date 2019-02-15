import numpy as np

class NaiveBayes():
	#multinominal NB model with laplace smoothing
	def __init__(self):
		self.data = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
					['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
					['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
					['stop', 'posting', 'stupid', 'worthless', 'garbage'],
					['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
					['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
		self.y = [0,1,0,1,0,1] #1 is abusive, 0 not
		self.p_w = {}
		self.p_c = {}
		self.vocabulary = []
		self.v_num = 0

	def train(self):
		self.label, p_c = np.unique(self.y, return_counts=True)
		self.p_c = dict(zip(self.label, np.log(p_c/len(self.y))))
		indexes = np.c_[np.array(self.y), np.arange(len(self.y))]

		self.vocabulary = np.unique([item for sublist in self.data for item in sublist])
		self.v_num = len(self.vocabulary)
		self.v_idx = dict(zip(self.vocabulary, np.arange(self.v_num)))

		for l in self.label:
			idxes = indexes[indexes[:,0]==l][:,1].astype(int)
			corpus = [self.data[idx] for idx in idxes]
			flatten = [item for sublist in corpus for item in sublist]
			self.p_w[l] = [np.log(1/(len(flatten)+self.v_num))]*self.v_num
			words, pwl = np.unique(flatten, return_counts=True)
			for w, p in zip(words, pwl):
				self.p_w[l][self.v_idx[w]] = np.log((p+1)/(len(flatten)+self.v_num))

	def predict(self, data):
		words, counts = np.unique(data, return_counts=True)
		v = np.zeros(self.v_num)

		for w, c in zip(words, counts):
			v[self.v_idx[w]] = c
		p = [0]*len(self.label)
		for i, l in enumerate(self.label):
			p[i] = v.dot(self.p_w[l])+self.p_c[l]
		return self.label[np.argmax(p)], p[np.argmax(p)]

nb = NaiveBayes()
nb.train()
for i in range(6):
	print(nb.predict(nb.data[i]))