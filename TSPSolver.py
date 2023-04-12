#!/usr/bin/python3


import time
import numpy as np
from BranchBound import BranchBound
from PriorityQueueImplementations import HeapPriorityQueue
from TSPClasses import *
import heapq
import itertools


class TSPSolver:
	def __init__( self, scenario = None):
		self._scenario = scenario

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution,
		time spent to find solution, number of permutations tried during search, the
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''
	# Time: The greedy algorithm used to initialize takes O(n^2). 
	#       This is because for each city, it has to look at the minimum nearby city, which takes O(n) for each city and O(n^2) overall
	# Space: The space is O(n) because we never have to store more than all of the cities in a list at any given time. Plus, this list is 1 dimensional.
	def greedy( self,time_allowance=60.0 ):
		start_time = time.time()
		cities = self._scenario.getCities()
		count = 0
		foundTour = False
		# maybe starting from the first city does not find a solution, so try different start cities
		while not foundTour and count < len(cities) and time.time()-start_time < time_allowance:
			curr = cities[count]
			visited = [curr]
			unvisited = cities[1:]
			while unvisited:
				closest = self.nearest_neighbor(curr, unvisited)
				if closest == None:
					break
				visited.append(closest)
				unvisited.remove(closest)
				curr = closest
			bssf = TSPSolution(visited)
			if bssf.cost < float('inf'):
				foundTour = True
			count += 1
				
		end_time = time.time()
		results = {}
		results['cost'] = bssf.cost
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results
	
	def nearest_neighbor(self, curr, cities):
		closest = None
		minv = float('inf')
		for city in cities:
			cost = curr.costTo(city)
			if cost < minv:
				minv = cost
				closest = city
		return closest



	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints:
		max queue size, total number of states created, and number of pruned states.</returns>
	'''
	# Time: The overall time is the time for the solve method in the solver (see discussion there, but we will go with O(n^3*2^n))
	#       plus the time to initialize the cost matrix and the initial bssf, which is no more than n^2. So overall O(n^3*2^n)
	# Space: The space is discussed in the solver.solve method, so O(nlogn*n^2)
	def branchAndBound( self, time_allowance=60.0 ):
		start_time = time.time()
		queue = HeapPriorityQueue()
		cities = self._scenario.getCities()
		costs = self.createCostMatrix(cities)
		paths = np.array([i for i in range(0, len(costs))])
		state = State(0, costs, 0, paths, paths, np.array([0]))
		states = {}
		states[0] = state
		solver = BranchBound(queue, time_allowance)
		bssf = self.getInitialBSSF(time_allowance)
		stats = solver.solve(states, bssf)
		end_time = time.time()
		sol = TSPSolution([cities[i] for i in stats.state.route] if stats.state else [])
		results = {}

		#if the final bssf equals the initial, then return 0
		cost = stats.bssf if stats.bssf != bssf else 0
		results['cost'] = cost
		results['time'] = end_time - start_time
		results['count'] = stats.num_solutions
		results['soln'] = sol
		results['max'] = stats.max_size
		results['total'] = stats.num_states
		results['pruned'] = stats.num_pruned

		return results
		

	# Time: The initial bssf I chose is the greedy algorithm which is O(n^2)
	# Space: The greedy algorithm has space O(n)
	def getInitialBSSF(self, time_allowance):
		results = self.greedy(time_allowance)
		return results['cost']

	# Time: This function loops through each cell in the 2d cost matrix, so it takes O(n^2) time
	# Space: The cost matrix is 2d of all of the cities, so O(n^2)
	def createCostMatrix(self, cities):
		costs = []
		for i in range(len(cities)):
			costs.append([])
			for j in range(len(cities)):
				cost = cities[i].costTo(cities[j])
				costs[i].append(cost)

		return costs

	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found during search, the
		best solution found.  You may use the other three field however you like.
		algorithm</returns>
	'''

'''
****************************************************** START EAX FUNCTIONS ********************************************************
'''	

	def EAX(self, distance_matrix, pop_size, max_iter, crossover_rate, mutation_rate):
		# Initialize population
		num_cities = distance_matrix.shape[0]
		population = []
		for i in range(pop_size):
			population.append(np.random.permutation(num_cities))

		# Main loop
		for i in range(max_iter):
			# Evaluate fitness of population
			fitness = []
			for p in population:
				fitness.append(total_distance(p, distance_matrix))
			
			# Selection
			parents = []
			for j in range(pop_size//2):
				p1, p2 = tournament_selection(population, fitness)
				parents.append(p1)
				parents.append(p2)
			
			# Crossover
			children = []
			for j in range(pop_size//2):
				if random.random() < crossover_rate:
					c1, c2 = EAX_crossover(parents[j], parents[j+1], distance_matrix)
					children.append(c1)
					children.append(c2)
			
			# Mutation
			for c in children:
				if random.random() < mutation_rate:
					c = swap_mutation(c)
			
			# Replacement
			population = replace_worst(population, children, fitness)
        
		# Find best solution
		best_fitness = float('inf')
		best_solution = None
		for p in population:
			p_fitness = total_distance(p, distance_matrix)
			if p_fitness < best_fitness:
				best_fitness = p_fitness
				best_solution = p
		
		return best_solution, best_fitness

	def total_distance(self, path, distance_matrix):
		return sum(distance_matrix[path[i], path[i+1]] for i in range(len(path)-1)) + distance_matrix[path[-1], path[0]]

	def tournament_selection(self, population, fitness, tournament_size=5):
		tournament = random.sample(range(len(population)), tournament_size)
		winner = tournament[0]
		for t in tournament[1:]:
			if fitness[t] < fitness[winner]:
				winner = t
		return population[winner], fitness[winner]

	def EAX_crossover(self, p1, p2, distance_matrix):
		edges = set()
		for i in range(len(p1)):
			edges.add((p1[i], p1[(i+1)%len(p1)]))
			edges.add((p2[i], p2[(i+1)%len(p2)]))
		
		cycles = []
		while edges:
			cycle = []
			current_edge = edges.pop()
			cycle.append(current_edge[0])
			next_node = current_edge[1]
			while next_node != cycle[0]:
				cycle.append(next_node)
				for edge in edges:
					if next_node == edge[0]:
						next_node = edge[1]
						edges.remove(edge)
						break
					elif next_node == edge[1]:
						next_node = edge[0]
						edges.remove(edge)
						break
			cycles.append(cycle)
		
		child = [-1]*len(p1)
		for i in range(len(cycles)):
			if i % 2 == 0:
				for j in cycles[i]:
					child[j] = p1[j]
			else:
				for j in cycles[i]:
					child[j] = p2[j]
		
		return child, p1[::-1]+p2[len(cycles[-1]):]+p2[:len(cycles[-1])]

	def swap_mutation(self, path):
		p1 = random.randint(0, len(path)-1)
		p2 = random.randint(0, len(path)-1)
		path[p1], path[p2] = path[p2], path[p1]
		return path

	def replace_worst(self, population, children, fitness):
		for c in children:
			worst_index = np.argmax(fitness)
			if fitness[worst_index] > total_distance(c, distance_matrix):
				population[worst_index] = c
				fitness[worst_index] = total_distance(c, distance_matrix)
		return population
'''
****************************************************** END EAX FUNCTIONS ********************************************************
'''	


class State:
	def __init__(self, index, cost, bound, out, inp, route):
		self.index = index
		self.out = out
		self.inp = inp
		self.route = route
		self.cost, b = self.reduce(cost)
		self.lower_bound = bound + b

	def key(self):
		return self.lower_bound / len(self.route)

	# Time: This function goes through each unvisited node and creates a new state.
	#       Creating a state reduces the cost matrix which is O(n^2) time, so doing this
	#			  for each route is O(n^3) time
	# Space: Each new state for each unvisited node contains its own copy matrix, so adding
	#        these up gives us a space of O(n^3)
	def expand(self):
		states = []
		# Looping through n nodes
		for i in range(len(self.out)):
			index = self.out[i]#._index
			if index == self.index:
				continue

			# Creating a copy of the array is O(n^2) time
			cost = np.ndarray.copy(self.cost)
			edge_cost = cost[self.index, index]

			# Each of these lines take O(n)
			cost[:, index] = float('inf')
			cost[self.index, :] = float('inf')
			out = np.delete(self.out, np.where(self.out == self.index))
			inp = np.delete(self.inp, np.where(self.inp == index))
			route = np.append(self.route, index)

			#When we create a new state, we reduce the cost matrix which is O(n^2) time
			state = State(index, cost, self.lower_bound + edge_cost, out, inp, route)
			states.append(state)
		return states
	def is_complete(self):
		return len(self.out) <= 1
	
	# Time: This function goes through each cell in the cost matrix to find the max and reduce
	#       values in the cells. This overall is O(n^2) time
	# Space: The space for this function never gets more than the space for the cost matrix, 
	#        or O(n^2)
	def reduce(self, cost):
		cost = np.array(cost)
		total = 0
		for i in range(len(self.out)):
			index = self.out[i]
			minv = min(cost[index, :])
			if minv == float('inf'):
				return cost, float('inf')
			cost[index, :] -= minv
			total += minv
		for i in range(len(self.inp)):
			index = self.inp[i]
			minv = min(cost[:, index])
			if minv == float('inf'):
				return cost, float('inf')
			cost[:, index] -= minv
			total += minv

		return cost, total
