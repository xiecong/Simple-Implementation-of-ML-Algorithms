import numpy as np
import matplotlib.pyplot as plt
'''
this code uses EA to find the weights of an mlp which learns the iris dataset
the mlp contains a hidden layer of six nuerons
tanh is used as activation function here. 
'''

def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
	eps = 1e-8
	x = np.exp(x)
	return x / (np.sum(x, axis=1).reshape(-1, 1) + eps)

class NN():
	def __init__(self, w1=None, b1=None, w2=None, b2=None):
		self.w1 = np.random.randn(4, 6) if w1 is None else w1
		self.b1 = np.random.randn(1, 6) if b1 is None else b1
		self.w2 = np.random.randn(6, 3) if w2 is None else w2
		self.b2 = np.random.randn(1, 3) if b2 is None else b2

	def loss(self, x, y): #using cross entropy as loss function
		eps = 1e-8
		h = self.predict(x)
		return -(np.multiply(y, np.log(h+eps)) + np.multiply((1 - y), np.log(1 - h+eps))).mean()

	def predict(self, x):
		o1 = tanh(x.dot(self.w1) + self.b1)
		return softmax(o1.dot(self.w2) + self.b2)

class EvolutionaryAlgorithm():
	def __init__(self):
		self.x = np.loadtxt("data/iris.data", delimiter = ',', usecols = (0,1,2,3), dtype = float)
		y = np.loadtxt("data/iris.data", delimiter = ',', usecols = (4), dtype = str).reshape(-1,1)
		class_mapping = {"b'Iris-setosa'":[1,0,0], "b'Iris-versicolor'":[0,1,0], "b'Iris-virginica'":[0,0,1]}
		self.y = np.array([class_mapping[yi[0]] for yi in y])
		self.pop_num = 100
		self.elitism_num = 20
		self.gen_num = 100
		self.mutate_rate = 0.1

	def cross_over(self, w1, w2):
		# CROSSOVER
		mask = np.random.uniform(0, 1, w1.shape)
		return (mask>0.5)*w1 + (mask<=0.5)*w2

	def mutate(self, w):
		mask = np.random.uniform(0, 1, w.shape)
		mutate_multiplier = np.random.randn(w.shape[0],w.shape[1])
		w += w*(mask<=self.mutate_rate)*mutate_multiplier

		mutate_range = 3 # prevent too large or small
		w[w > mutate_range] = mutate_range
		w[w < -mutate_range] = -mutate_range

	def evolve(self, old_pop):
		eps = 1e-8
		fitness = np.array([1 / (p.loss(self.x, self.y)+eps) for p in old_pop])
		fitness = fitness/fitness.sum()
		top = np.argsort(fitness)[-1:-self.elitism_num-1:-1]
		new_pop = [old_pop[idx] for idx in top]
		for i in range(self.pop_num - self.elitism_num):
			# SELECTION by probabilities (fitness)
			idxes = np.random.choice(self.pop_num, 2, p=fitness)
			a = old_pop[idxes[0]]
			b = old_pop[idxes[1]]
			# CROSSOVER
			w1 = self.cross_over(a.w1, b.w1)
			b1 = self.cross_over(a.b1, b.b1)
			w2 = self.cross_over(a.w2, b.w2)
			b2 = self.cross_over(a.b2, b.b2)
			# MUTATION
			self.mutate(w1)
			self.mutate(b1)
			self.mutate(w2)
			self.mutate(b2)
			new_pop.append(NN(w1,b1,w2,b2))

		loss = [p.loss(self.x, self.y) for p in new_pop]
		return new_pop, loss

	def run(self):
		population = [NN() for _ in range(self.pop_num)]
		losslog = []
		for i in range(self.gen_num):
			population, loss = self.evolve(population)
			losslog.append([max(loss), np.mean(loss), min(loss)])
			#print("step{} max:{} min:{} mean:{}".format(i, max(loss), min(loss), np.mean(loss)))
		best = np.argmin(loss)
		print([np.argmax(o) for o in population[best].predict(self.x)])
		losslog = np.array(losslog)
		plt.plot(losslog[:,0])
		plt.plot(losslog[:,1])
		plt.plot(losslog[:,2])
		plt.legend(('max','mean','best'),loc='best')
		plt.title('loss over generation')
		plt.show()

ea = EvolutionaryAlgorithm()
ea.run()