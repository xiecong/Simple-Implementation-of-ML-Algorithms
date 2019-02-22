import numpy as np
from sklearn import datasets
import re

def clean_text(documents, stop_words):
	text = []
	for doc in documents:
		letters_only = re.sub("[^a-zA-Z]", " ", doc)
		words = letters_only.lower().split()             
		text.append([w for w in words if not w in stop_words])
	return text

class NaiveBayes():
	#multinominal NB model with laplace smoothing
	#guassian can be used for numerical
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.p_w = {}
		self.p_c = {}
		self.vocabulary = []
		self.v_num = 0

	def fit(self):
		self.label, p_c = np.unique(self.y, return_counts=True)
		self.p_c = dict(zip(self.label, np.log(p_c/len(self.y))))
		indexes = np.c_[np.array(self.y), np.arange(len(self.y))]

		self.vocabulary = np.unique([item for sublist in self.x for item in sublist])
		self.v_num = len(self.vocabulary)
		print("vocabulary length {}".format(self.v_num))
		self.v_idx = dict(zip(self.vocabulary, np.arange(self.v_num)))

		print("start fitting")
		for l in self.label:
			idxes = indexes[indexes[:,0]==l][:,1].astype(int)
			corpus = [self.x[idx] for idx in idxes]
			flatten = [item for sublist in corpus for item in sublist]
			self.p_w[l] = [np.log(1/(len(flatten)+self.v_num))]*self.v_num
			words, pwl = np.unique(flatten, return_counts=True)
			for w, p in zip(words, pwl):
				self.p_w[l][self.v_idx[w]] = np.log((p+1)/(len(flatten)+self.v_num))

	def predict(self, x):
		p = [0]*len(self.label)
		for i, l in enumerate(self.label):
			p[i] = self.p_c[i]
			for w in x:
				p[i] += self.p_w[i][self.v_idx[w]]
		return self.label[np.argmax(p)]

def main():
	stop_words = set(["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"])
	data = datasets.fetch_20newsgroups()
	x = clean_text(data.data, stop_words)
	y = data.target
	nb = NaiveBayes(x, y)
	nb.fit()
	print("predicting")
	print(sum(nb.predict(xi) == y[i] for i, xi in enumerate(x))/len(x))

if __name__ == "__main__":
    main()