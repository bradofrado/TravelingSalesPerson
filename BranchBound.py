import time
import numpy as np

class BranchBound:
	def __init__(self, queue, time_allowance):
		self.time_allowance = time_allowance
		self.queue = queue
	def solve(self, states, bssf):
		start_time = time.time()
		stats = BranchStats(bssf, 1)
		self.states = states
		self.queue.insert(self.states[0], self.states[0].key())
		while not self.queue.empty() and time.time()-start_time < self.time_allowance:
			if len(self.queue) > stats.max_size:
				stats.max_size = len(self.queue)
			pBig = self.queue.delete_min()
			#prune
			if pBig.lower_bound > stats.bssf:
				stats.num_pruned += 1
				continue
			pks = pBig.expand()
			stats.num_states += len(pks)
			for p in pks:
				if p.is_complete():
					stats.bssf = p.lower_bound
					stats.state = p
					stats.num_solutions += 1
				elif p.lower_bound < stats.bssf:
					self.queue.insert(p, p.key())
				else:
					stats.num_pruned += 1
		end_time = time.time()
		stats.time = end_time - start_time
		#prune anything on the after completed the time_allowance
		while not self.queue.empty():
				p = self.queue.delete_min()
				if p.lower_bound > stats.bssf:
					stats.num_pruned += 1	
		return stats
	

class BranchStats:
	def __init__(self, bssf, max_size):
		self.max_size = max_size
		self.bssf = bssf
		self.state = None
		self.num_states = max_size
		self.num_pruned = 0
		self.num_solutions = 0
		self.time = 0
		pass
		
                  
                  
        
        
		