import numpy as np
from pandas import read_csv
# an example of asia bayesian net:
# https://www.eecis.udel.edu/~shatkay/Course/papers/Lauritzen1988.pdf


class BayesianNet(object):
	def __init__(self, names, edges, tables=None):
		self.n_nodes = len(names)
		if tables is None: tables = [[0]] * self.n_nodes
		self.nodes = [{'name': name, 'table': np.array(table)} for name, table in zip(names, tables)]
		self.name2idx = {k: v for v, k in enumerate(names)}
		self.graph = np.zeros((self.n_nodes, self.n_nodes))
		for edge in edges:
			self.graph[self.name2idx[edge[1]], self.name2idx[edge[0]]] = 1
		self.binary = np.array([1 << self.n_nodes - 1 - i for i in range(self.n_nodes)])

	def fit(self, data):
		data_size = len(data)
		for i, node in enumerate(self.nodes):
			table = []
			parents = self.graph[i]==1
			marginal = data[:, parents]
			index = np.zeros(data.shape[0])
			if marginal.shape[1] > 0:
				index = (marginal * self.binary[-marginal.shape[1]:]).sum(axis=1)
			for j in range(2**parents.sum()):
				table.append(data[(index == j), i].sum() / (index == j).sum())
			node['table'] = np.array(table)

	def joint_p(self, values):
		p = 1
		for i in range(self.n_nodes):
			index = 0
			parents = self.graph[i]==1
			if parents.sum() > 0:
				index = np.dot(values[parents], self.binary[-parents.sum():])
			p *= (1 - values[i]) + (2 * values[i] - 1) * self.nodes[i]['table'][int(index)]
		return p

	def marginal_p(self, condition):
		p = 0
		values = -np.ones(self.n_nodes)
		for v in condition:
			values[self.name2idx[v[1]]] = int(v[0] != '~')
		mask = np.arange(self.n_nodes)[(values==-1)]
		n_unkowns = self.n_nodes - len(condition)
		for i in range(2**n_unkowns):
			values[mask] = np.array([int(x) for x in '{:0{size}b}'.format(i, size=n_unkowns)])
			p += self.joint_p(values)
		return p

	def query(self, v, condition):
		p_pos = self.marginal_p([f'+{v}'] + condition) / self.marginal_p(condition)
		return [1 - p_pos, p_pos]

def get_asia_data(url):
	return read_csv(url).apply(lambda x: x == 'yes').astype(int).values


def main():
	names = 'ATSLBEXD'
	edges = ['AT', 'SL', 'SB', 'TE', 'LE', 'BD', 'EX', 'ED']
	#tables = [[0.01], [0.01, 0.05], [0.5], [0.01, 0.1], [0.3, 0.6], [0, 1, 1, 1], [0.05, 0.98], [0.1, 0.7, 0.8, 0.9]]
	bn = BayesianNet(list(names), edges)  # also can use predefined conditional tables
	asia_url = 'http://www.ccd.pitt.edu/wiki/images/ASIA10k.csv'
	bn.fit(get_asia_data(asia_url))
	print(bn.nodes)
	for condition in [[], ['+A', '~S'], ['+A', '~S', '~D', '+X']]:
		for c in ['T', 'L', 'B', 'E']:
			print('p({}|{})={}'.format(c, ','.join(condition), bn.query(c, condition)))


if __name__ == "__main__":
	main()