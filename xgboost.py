import numpy as np
# importance for each feature
class XGBoost(object):
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def fit(self):
		pass

	def predict(self):
		pass

def main():
	data = datasets.load_boston()
	x = data.data
	y = data.target
	xgboost = XGBoost(x, y)

if __name__ == "__main__":
	main()