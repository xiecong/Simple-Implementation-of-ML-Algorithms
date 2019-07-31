import numpy as np
from sklearn.datasets import load_iris
# TO DO: add sample weights


def weighted_histo(y, w):
	y_unique = np.unique(y)
	return [w[y==yi].sum() for yi in y_unique] / w.sum()

def entropy(y, w):
	p = weighted_histo(y, w)
	return -np.sum(np.multiply(p, np.log2(p+1e-8)))

def impurity(y, w):
	return 1 - np.sum(np.square(weighted_histo(y, w)))

def variance(y, w):
	mu = y.dot(w) / len(y)
	return np.square(y - mu).dot(w) / sum(w)


class DecisionTree(object):
	def __init__(self, metric_type, depth, regression=False):		
		metrics = {'Info gain': entropy, 'Gini impurity':impurity, 'Variance': variance}
		self.regression = regression
		self.metric = metrics[metric_type]
		self.tree = {}
		self.depth = depth
		self.gain_threshold = 1e-8

	def split_gain(self, p_score, l_y, r_y, l_w, r_w):
		total_w = sum(l_w) + sum(r_w)
		return p_score - (self.metric(l_y, l_w)*sum(l_w) + self.metric(r_y, r_w)*sum(r_w)) / total_w

	def print_tree(self, node=None, depth=0):
		if node is None:
			node = self.tree
		if 'f_id' in node:
			print('{}[X{} < {}]'.format(depth*' ', (node['f_id']+1), node['value']))
			self.print_tree(node['left'], depth+1)
			self.print_tree(node['right'], depth+1)
		else:
			print('{}{}'.format(depth*' ', node))

	def gen_leaf(self, y, w):
		if not self.regression:
			weighted_sum = [w[y==li].sum() for li in self.labels]
			node = dict(zip(self.labels, weighted_sum))
			node['label'] = self.labels[np.argmax(weighted_sum)]
		else:
			node = {'label': y.dot(w) / sum(w)}
		return node

	def split(self, x, y, w, depth):
		if(depth >= self.depth): return self.gen_leaf(y, w)
		p_score = self.metric(y, w)
		max_gain, f_id, value = self.gain_threshold, -1, 0, 
		splt_l_x, splt_r_x, splt_l_y, splt_r_y, splt_l_w, splt_r_w = None, None, None, None, None, None
		for f in self.feature_set:
			split_values = np.unique(x[:,f].round(decimals=4))
			for split_value in split_values:
				l_idx, r_idx = x[:,f]<split_value, x[:,f]>=split_value
				l_x, l_y, l_w = x[l_idx], y[l_idx], w[l_idx]
				r_x, r_y, r_w = x[r_idx], y[r_idx], w[r_idx]
				if(len(l_x)*len(r_x)==0):
					continue
				gain = self.split_gain(p_score, l_y, r_y, l_w, r_w)
				if gain > max_gain:
					max_gain, f_id, value = gain, f, split_value
					splt_l_x, splt_l_y, splt_l_w = l_x, l_y, l_w
					splt_r_x, splt_r_y, splt_r_w = r_x, r_y, r_w
		if f_id != -1:
			self.importance[f_id] += max_gain * sum(w)
			return {
				'f_id': f_id,
				'value': value,
				'left': self.split(splt_l_x, splt_l_y, splt_l_w, depth+1),
				'right': self.split(splt_r_x, splt_r_y, splt_r_w, depth+1)
			}
		else: return self.gen_leaf(y, w)

	def fit(self, x, y, w=None, feature_set=None):
		self.labels = np.unique(y)
		self.feature_set = np.arange(x.shape[1]) if feature_set is None else feature_set
		self.importance = np.zeros(x.shape[1])
		self.tree = self.split(x, y, np.ones(x.shape[0]) if w is None else w, 0)
		self.importance /= len(x)

	def predict(self, x):
		return np.array([self.predict_sample(xi) for xi in x])

	def predict_sample(self, sample, node=None):
		if node is None:
			node = self.tree
		if 'f_id' in node:
			child = node['left'] if(sample[node['f_id']] < node['value']) else node['right']
			return self.predict_sample(sample, child)
		else:
			return node['label']

	def get_importance(self):
		return self.importance


def main():
	data = load_iris()
	test_ratio = 0.2
	test_split = np.random.uniform(0, 1, len(data.data))
	train_x, test_x = data.data[test_split >= test_ratio], data.data[test_split < test_ratio]
	train_y, test_y = data.target[test_split >= test_ratio], data.target[test_split < test_ratio]

	dt = DecisionTree(metric_type='Gini impurity', depth=4)
	dt.fit(train_x, train_y)
	dt.print_tree()
	print(dt.importance)
	print(sum(dt.predict(train_x)==train_y) / len(train_x))
	print(sum(dt.predict(test_x)==test_y) / len(test_x))


if __name__ == "__main__":
    main()