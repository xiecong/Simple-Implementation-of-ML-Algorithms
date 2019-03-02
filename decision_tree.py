import numpy as np
from sklearn import datasets

def entropy(data):
	(values, counts) = np.unique(data[:,-1],return_counts=True)
	p = counts/len(data)
	return -np.sum(np.multiply(p, np.log2(p+1e-8)))

def impurity(data):	
	(values, counts) = np.unique(data[:,-1],return_counts=True)
	p = counts/len(data)
	return 1 - np.sum(np.square(p))

def variance(data):
	avg = data[:,-1].mean()
	return np.square(avg - data[:,-1]).sum()

class DecisionTree(object):
	def __init__(self, data, metric_type, depth, regression=False, feats=None):		
		metrics = {'Info gain': entropy, 
				   'Gini impurity':impurity, 
				   'Variance': variance
				   }
		self.data = data
		self.labels = np.unique(data[:,-1])
		self.regression = regression
		self.metric = metrics[metric_type]
		self.tree = {}
		self.feat_num = data.shape[1] - 1
		self.features = np.arange(self.feat_num) if feats is None else feats
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

	def assign(self, data, f_id, value):
		# split
		l_child = [d for d in data if float(d[f_id]) < value]
		r_child = [d for d in data if float(d[f_id]) >= value]
		return np.array(l_child), np.array(r_child)

	def gen_leaf(self, data):
		if not self.regression:
			(values, counts) = np.unique(data[:,-1],return_counts=True)
			node = dict(zip(values, counts))
			node['label'] = values[np.argmax(counts)]
		else:
			node = {'label': data[:,-1].mean()}
		return node

	def split(self, data, depth):
		if(depth >= self.depth): return self.gen_leaf(data)
		p_score = self.metric(data)
		max_gain, f_id, value, splt_l, splt_r = self.gain_threshold, -1, 0, None, None
		for f in self.features:
			for d in data:
				l_child, r_child = self.assign(data, f, float(d[f]))
				if(len(l_child)*len(r_child)==0):
					continue
				gain = self.split_gain(p_score, l_child, r_child)
				if gain > max_gain:
					max_gain = gain
					f_id = f
					value = float(d[f])
					splt_l, splt_r = l_child, r_child
		if f_id != -1:
			return {'f_id': f_id, 
					'value': value,
					'left': self.split(splt_l, depth+1),
					'right': self.split(splt_r, depth+1)
				   }
		else: return self.gen_leaf(data)

	def fit(self):
		self.tree = self.split(self.data, 0)

	def predict(self, sample, node=None):
		if node is None:
			node = self.tree
		if 'f_id' in node:
			child = node['left'] if(sample[node['f_id']] < node['value']) else node['right']
			return self.predict(sample, child)
		else:
			return node['label']

def main():
	data = datasets.load_iris()
	x = data.data
	y = data.target
	dt = DecisionTree(data=np.c_[x, y], metric_type='Gini impurity', depth=4)
	dt.fit()
	dt.print_tree()
	print(sum(dt.predict(xi)==y[i] for i, xi in enumerate(x)))

if __name__ == "__main__":
    main()