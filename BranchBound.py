import numpy as np

class BranchBound:
	def __init__(self, queue, time_allowance):
		self.time_allowance = time_allowance
		self.queue = queue
	def solve(self, states, bssf):
		self.states = states
		self.queue.insert(self.states[0].index, self.states[0].lower_bound)
		while not self.queue.empty():
			pBig = self.states[self.queue.delete_min()]
			pks = pBig.expand()
			for p in pks:
				if p.is_complete():
					bssf = p.lower_bound
				elif p.lower_bound < bssf:
					self.queue.insert(p.index, p.lower_bound)
				self.states[p.index] = p
		return bssf
		
                  
                  
        
        
		