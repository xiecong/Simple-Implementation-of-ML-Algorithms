import numpy as np

class LogisticRegression():
	def __init__(self, metric):
		self.x = np.loadtxt("iris.data", delimiter = ',', usecols = (0,1,2,3), dtype = float)
		self.y = np.loadtxt("iris.data", delimiter = ',', usecols = (4), dtype = str)
		# ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

	def train():
		pass

	def predict():
		pass

