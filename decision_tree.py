import numpy as np
from sklearn import datasets

def entropy(p):
	return -np.sum(np.multiply(p, np.log2(p+1e-8)))

def impurity(p):
	return 1 - np.sum(np.square(p))

class DecisionTree(object):
	def __init__(self, data, metric_type, depth, feats=None):		
		metrics = {'Info gain': entropy, 'Gini impurity':impurity}
		self.data = data
		self.labels = np.unique(data[:,-1])
		self.metric = metrics[metric_type]
		self.tree = {}
		self.feat_num = data.shape[1] - 1
		self.features = np.arange(self.feat_num) if feats is None else feats
		self.depth = depth
		self.gain_threshold = 1e-8

	def get_score(self, data):
		(values, counts) = np.unique(data[:,-1],return_counts=True)
		return self.metric(counts/len(data))

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
		(values, counts) = np.unique(data[:,-1],return_counts=True)
		node = dict(zip(values, counts))
		node['label'] = values[np.argmax(counts)]
		return node

	def split(self, data, p_score, depth):
		if(depth >= self.depth):
			return self.gen_leaf(data)
		data_num = len(data)
		score, f_id, value, splt = float('inf'), -1, 0, None
		for f in self.features:
			for d in data:
				l_child, r_child = self.assign(data, f, float(d[f]))
				if(len(l_child)*len(r_child)==0):
					continue
				l_score = self.get_score(l_child)
				r_score = self.get_score(r_child)
				c_score = l_score*len(l_child)/data_num + r_score*len(r_child)/data_num
				if c_score < score:
					score = c_score
					f_id = f
					value = float(d[f])
					splt = {'l_child':l_child,
							'l_score':l_score,
							'r_child':r_child,
							'r_score':r_score}
		if f_id != -1 and p_score - score > self.gain_threshold:
			return {'f_id': f_id, 
					'value': value,
					'left': self.split(splt['l_child'], splt['l_score'], depth+1),
					'right': self.split(splt['r_child'], splt['r_score'], depth+1)
				   }
		else: return self.gen_leaf(data)

	def fit(self):
		score = self.get_score(self.data)
		self.tree = self.split(self.data, score, 0)

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
	dt = DecisionTree(np.c_[x, y],'Gini impurity', 4)
	dt.fit()
	dt.print_tree()
	print(sum(dt.predict(xi)==y[i] for i, xi in enumerate(x)))

if __name__ == "__main__":
    main()