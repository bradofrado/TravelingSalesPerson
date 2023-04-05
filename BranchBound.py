import time
import numpy as np

class BranchBound:
	def __init__(self, queue, time_allowance):
		self.time_allowance = time_allowance
		self.queue = queue
	# Time: Assuming nothing gets pruned, we have an overall complexity of O(n^3*n!). 
	#       This is because each pass through is O(n^3) (expanding the children) and we do this for n! states. 
	#       However, because we are pruning, this takes off a lot of time. Even if we were expanding just two states at each additional level and the rest pruning, 
	#       then we would be at O(n^3*2^n).
	# Space: The max number of states on the priority queue does not grow by anything more than nlogn. 
	# 			 Even though each state as a potential of holding n^3 cities when it is expanded, 
	#        we are only ever expanding one state at a time and these new states add to our overall queue, 
	#        so really the size of the algorithm is the size of the queue * the cost matrix, which is O(nlogn*n^2)
	def solve(self, states, bssf):
		start_time = time.time()
		stats = BranchStats(bssf, 1)
		self.states = states

		#This insert statement is O(1) because there is nothing on the queue
		self.queue.insert(self.states[0], self.states[0].key())
		# Worst case, we loop through each state which is n! times n^3 or O(n^3*n!)
		while not self.queue.empty() and time.time()-start_time < self.time_allowance:
			if len(self.queue) > stats.max_size:
				stats.max_size = len(self.queue)
			# Deleting from the priotiry queue is O(logn) because it has to traverse the 
			# binary tree of depth log(n)
			pBig = self.queue.delete_min()
			if pBig.lower_bound > stats.bssf:
				stats.num_pruned += 1
				continue
			# expanding the state takes O(n^3) time
			pks = pBig.expand()
			stats.num_states += len(pks)

			# We loop through each state. Worst case, we are inserting each state in the queu
			# which takes O(logn), so overall this loop will take O(nlogn)
			for p in pks:
				if p.is_complete():
					stats.bssf = p.lower_bound
					stats.state = p
					stats.num_solutions += 1
				elif p.lower_bound < stats.bssf:
					# Inserting into the priotiry queue is O(logn) because it has to traverse the 
					# binary tree of depth log(n)
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
		
                  
                  
        
        
		