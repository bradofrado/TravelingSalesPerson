import numpy as np

class BranchBound:
	def __init__(self, queue, time_allowance):
		self.time_allowance = time_allowance
		self.queue = queue
	def solve(self, states):
		self.states = states
		self.queue.insert(self.states[0].index, self.states[0].lower_bound)
		bssf = self.getInitialBSSF(states)
		i = 0
		while not self.queue.empty():
			if i == 5:
				i = 0
			pBig = self.states[self.queue.delete_min()]
			pks = pBig.expand()
			for p in pks:
				if p.is_complete():
					bssf = p.lower_bound
				elif p.lower_bound < bssf:
					self.queue.insert(p.index, p.lower_bound)
				self.states[p.index] = p
			i += 1
		return bssf
	
	def getInitialBSSF(self, states):
		return float('inf')
		pass

class State:
	def __init__(self, index, cost, bound, path):
		self.index = index
		self.path = path
		self.cost, b = self.reduce(cost)
		self.lower_bound = bound + b

	def expand(self):
		states = []
		for i in range(len(self.path)):
			cost = np.ndarray.copy(self.cost)
			index = self.path[i]
			cost[:, self.path[i]] = float('inf')
			cost[self.path[i], :] = float('inf')
			path = np.delete(self.path, i)
			state = State(index, cost, self.lower_bound, path)
			states.append(state)
		return states
	def is_complete(self):
		return len(self.path) == 0
	def reduce(self, cost):
		cost = np.array(cost)
		total = 0
		for i in range(len(self.path)):
			minv = min(cost[self.path[i], :])
			if minv == float('inf'):
				return cost, float('inf')
			cost[self.path[i], :] -= minv
			total += minv
		for i in range(len(self.path)):
			minv = min(cost[:, self.path[i]])
			if minv == float('inf'):
				return cost, float('inf')
			cost[:, self.path[i]] -= minv
			total += minv

		return cost, total

		
                  
                  
        
        
		