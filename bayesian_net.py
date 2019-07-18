import numpy as np
import pandas as pd
# an example of asia bayesian net:
# https://www.eecis.udel.edu/~shatkay/Course/papers/Lauritzen1988.pdf

class BayesianNet(object):
	def __init__(self, names, edges, tables=None):
		if tables is None: tables = [[0]] * len(names)
		self.nodes = [{'name': name, 'table': np.array(table)} for name, table in zip(names, tables)]
		self.name2idx = {k: v for v, k in enumerate(names)}
		self.graph = np.zeros((8,8))
		for edge in edges:
			self.graph[self.name2idx[edge[1]], self.name2idx[edge[0]]] = 1

	def fit(self, data):
		data_size = len(data)
		for i, node in enumerate(self.nodes):
			table = []
			marginal = data[:,(self.graph[i]==1)]
			index = np.zeros(marginal.shape[0])
			for j in range(marginal.shape[1]):
				index = index * 2 + marginal[:, j]
			for j in range(2**(self.graph[i]==1).sum()):
				table.append(data[(index == j), i].sum() / (index == j).sum())
			node['table'] = np.array(table)

	def query(self, v, condition):
		return 0

def get_asia_data():
	url = 'http://www.ccd.pitt.edu/wiki/images/ASIA10k.csv'
	return pd.read_csv(url).apply(lambda x: x == 'yes').astype(int).values

def main():
	asia_data = get_asia_data()
	names = 'ATSLBEXD'
	edges = ['AT', 'SL', 'SB', 'TE', 'LE', 'BD', 'EX', 'ED']
	#tables = [[0.01], [0.01, 0.05], [0.5], [0.01, 0.1], [0.3, 0.6], [0, 1, 1, 1], [0.05, 0.98], [0.1, 0.7, 0.8, 0.9]]
	bn = BayesianNet(list(names), edges)
	bn.fit(asia_data)
	print(bn.nodes)

	for condition in [[], [{'A': 1}, {'S': 0}], [{'A': 1}, {'S': 0}, {'D': 0}, {'X': 1}]]:
		for c in ['T', 'L', 'B']:
			print("p({}=1|{})={}".format(c, condition, bn.query({c: 1}, condition)))

if __name__ == "__main__":
	main()