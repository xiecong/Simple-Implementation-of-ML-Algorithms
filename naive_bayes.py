import numpy as np
from sklearn.datasets import fetch_20newsgroups
import re


def tokenize(documents, stop_words):
	text = []
	for doc in documents:
		letters_only = re.sub("[^a-zA-Z]", " ", doc)
		words = letters_only.lower().split()             
		text.append([w for w in words if not w in stop_words])
	return np.array(text)


class NaiveBayes(object):
	#multinominal NB model with laplace smoothing
	#guassian can be used for numerical
	def __init__(self):
		self.p_w = {}
		self.p_c = {}
		self.vocabulary = []
		self.v_num = 0

	def fit(self, x, y):
		n_data = len(y)
		self.label, p_c = np.unique(y, return_counts=True)
		self.p_c = dict(zip(self.label, np.log(p_c/n_data)))
		indexes = np.c_[np.array(y), np.arange(n_data)]

		self.vocabulary = np.unique([item for sublist in x for item in sublist])
		self.v_num = len(self.vocabulary)
		print("vocabulary length {}".format(self.v_num))
		self.v_idx = dict(zip(self.vocabulary, np.arange(self.v_num)))

		print("start fitting")
		for l in self.label:
			idxes = indexes[indexes[:,0]==l][:,1].astype(int)
			corpus = [x[idx] for idx in idxes]
			flatten = [item for sublist in corpus for item in sublist]
			self.p_w[l] = [np.log(1/(len(flatten)+self.v_num))]*self.v_num
			words, pwl = np.unique(flatten, return_counts=True)
			for w, p in zip(words, pwl):
				self.p_w[l][self.v_idx[w]] = np.log((p+1)/(len(flatten)+self.v_num))

	def predict(self, x):
		return np.array([self.predict_sample(xi) for xi in x])

	def predict_sample(self, x):
		eps = 1 / self.v_num
		p = [self.p_c[i] + sum(self.p_w[i][self.v_idx[w]] if w in self.v_idx.keys() else eps for w in x) for i in range(len(self.label))]
		return self.label[np.argmax(p)]


def main():
	stop_words = set(["i", "me", "my", "myself", "we", "our", "ours", "ourselves",
		"you", "your", "yours", "yourself", "yourselves", "he", "him", "his",
		"himself", "she", "her", "hers", "herself", "it", "its", "itself", "they",
		"them", "their", "theirs", "themselves", "what", "which", "who", "whom",
		"this", "that", "these", "those", "am", "is", "are", "was", "were", "be",
		"been", "being", "have", "has", "had", "having", "do", "does", "did",
		"doing", "a", "an", "the", "and", "but", "if", "or", "because", "as",
		"until", "while", "of", "at", "by", "for", "with", "about", "against",
		"between", "into", "through", "during", "before", "after", "above", "below",
		"to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
		"again", "further", "then", "once", "here", "there", "when", "where", "why",
		"how", "all", "any", "both", "each", "few", "more", "most", "other", "some",
		"such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
		"very", "s", "t", "can", "will", "just", "don", "should", "now"])
	data = fetch_20newsgroups()
	x = tokenize(data.data, stop_words)
	y = data.target

	test_ratio = 0.2
	test_split = np.random.uniform(0, 1, len(x))
	train_x = x[test_split >= test_ratio]
	test_x = x[test_split < test_ratio]
	train_y = y[test_split >= test_ratio]
	test_y = y[test_split < test_ratio]

	nb = NaiveBayes()
	nb.fit(train_x, train_y)
	print("predicting")
	print(sum(nb.predict(train_x) == train_y)/train_x.shape[0])
	print(sum(nb.predict(test_x) == test_y)/test_y.shape[0])


if __name__ == "__main__":
    main()