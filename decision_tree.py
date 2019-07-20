import numpy as np
from sklearn.datasets import load_iris


def entropy(data):
	(values, counts) = np.unique(data[:,-1],return_counts=True)
	p = counts/len(data)
	return -np.sum(np.multiply(p, np.log2(p+1e-8)))

def impurity(data):	
	(values, counts) = np.unique(data[:,-1],return_counts=True)
	p = counts/len(data)
	return 1 - np.sum(np.square(p))

def variance(data):
	return data[:,-1].var()


class DecisionTree(object):
	def __init__(self, metric_type, depth, regression=False):		
		metrics = {'Info gain': entropy, 'Gini impurity':impurity, 'Variance': variance}
		self.regression = regression
		self.metric = metrics[metric_type]
		self.tree = {}
		self.depth = depth
		self.gain_threshold = 1e-8

	def split_gain(self, p_score, l_child, r_child):
		data_num = len(l_child)+len(r_child)
		return p_score - self.metric(l_child)*len(l_child)/data_num\
			- self.metric(r_child)*len(r_child)/data_num

	def print_tree(self, node=None, depth=0):
		if node is None:
			node = self.tree
		if 'f_id' in node:
			print('{}[X{} < {}]'.format(depth*' ', (node['f_id']+1), node['value']))
			self.print_tree(node['left'], depth+1)
			self.print_tree(node['right'], depth+1)
		else:
			print('{}{}'.format(depth*' ', node))

	def gen_leaf(self, data):
		if not self.regression:
			(values, counts) = np.unique(data[:,-1],return_counts=True)
			node = dict(zip(values, counts))
			node['label'] = values[np.argmax(counts)]
		else:
			node = {'label': data[:,-1].mean()}
		return node

	def split(self, data, depth, n_data):
		if(depth >= self.depth): return self.gen_leaf(data)
		p_score = self.metric(data)
		max_gain, f_id, value, splt_l, splt_r = self.gain_threshold, -1, 0, None, None
		for f in self.feature_set:
			split_values = np.unique(data[:,f].round(decimals=4))
			for split_value in split_values:
				l_child = data[np.nonzero(data[:,f]<split_value)]
				r_child = data[np.nonzero(data[:,f]>=split_value)]
				if(len(l_child)*len(r_child)==0):
					continue
				gain = self.split_gain(p_score, l_child, r_child)
				if gain > max_gain:
					max_gain = gain
					f_id = f
					value = split_value
					splt_l, splt_r = l_child, r_child
		if f_id != -1:
			self.importance[f_id] += max_gain * data.shape[0] / n_data
			return {'f_id': f_id,
					'value': value,
					'left': self.split(splt_l, depth+1, n_data),
					'right': self.split(splt_r, depth+1, n_data)
				   }
		else: return self.gen_leaf(data)

	def fit(self, x, y, feature_set=None):
		self.labels = np.unique(y)
		self.feature_set = np.arange(x.shape[1]) if feature_set is None else feature_set
		self.importance = np.zeros(x.shape[1])
		self.tree = self.split(np.c_[x, y], 0, len(x))

	def predict(self, sample, node=None):
		if node is None:
			node = self.tree
		if 'f_id' in node:
			child = node['left'] if(sample[node['f_id']] < node['value']) else node['right']
			return self.predict(sample, child)
		else:
			return node['label']

	def get_importance(self):
		return self.importance


def main():
	data = load_iris()
	x = data.data
	y = data.target

	test_ratio = 0.3
	test_split = np.random.uniform(0, 1, len(x))
	train_x = x[test_split >= test_ratio]
	test_x = x[test_split < test_ratio]
	train_y = y[test_split >= test_ratio]
	test_y = y[test_split < test_ratio]

	dt = DecisionTree(metric_type='Gini impurity', depth=4)
	dt.fit(train_x, train_y)
	dt.print_tree()
	print(dt.importance)
	print(sum(dt.predict(xi)==yi for xi, yi in zip(train_x, train_y)) / len(train_x))
	print(sum(dt.predict(xi)==yi for xi, yi in zip(test_x, test_y)) / len(test_x))


if __name__ == "__main__":
    main()