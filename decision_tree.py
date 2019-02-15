import numpy as np

def entropy(dist):
	return -np.sum(np.multiply(dist, np.log2(dist+1e-8)))

def impurity(dist):
	return 1 - np.sum(np.square(dist))

class DecisionTree():
	def __init__(self, metric_type, depth):
		metrics = {'Info gain': entropy, 'Gini impurity':impurity}
		x = np.loadtxt("data/iris.data", delimiter = ',', usecols = (0,1,2,3), dtype = float)
		y = np.loadtxt("data/iris.data", delimiter = ',', usecols = (4), dtype = str)
		self.data = np.c_[x, y]
		self.labels = np.unique(y)
		self.metric = metrics[metric_type]
		self.tree = {}
		self.feat_num = self.data.shape[1] - 1
		self.depth = depth

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
		return dict(zip(values, counts))

	def split(self, data, p_score, depth):
		if(p_score < 1e-8 or depth >= self.depth):
			return self.gen_leaf(data)
		score, f_id, value, splt = float('inf'), -1, 0, None
		for f in range(self.feat_num):
			for d in data:
				l_child, r_child = self.assign(data, f, float(d[f]))
				if(len(l_child)*len(r_child)==0):
					continue
				l_score = self.get_score(l_child)
				r_score = self.get_score(r_child)
				#print("{}, {}".format(l_score,r_score))
				if l_score + r_score < score:
					score = l_score + r_score
					f_id = f
					value = float(d[f])
					splt = {'l_child':l_child,
							'l_score':l_score,
							'r_child':r_child,
							'r_score':r_score}
		if f_id != -1:
			return {'f_id': f_id, 
					'value': value,
					'left': self.split(splt['l_child'], splt['l_score'], depth+1),
					'right': self.split(splt['r_child'], splt['r_score'], depth+1)
				   }
		return self.gen_leaf(data)

	def train(self):
		score = self.get_score(self.data)
		self.tree = self.split(self.data, score, 0)

	def predict(self, sample, node=None):
		if node is None:
			node = self.tree
		if 'f_id' in node:
			child = node['left'] if(sample[node['f_id']] < node['value']) else node['right']
			return self.predict(sample, child)
		else:
			return node

dt = DecisionTree('Info gain', 3)
dt.train()
dt.print_tree()
print(dt.predict([6.7,3.0,5.2,2.3]))