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
	def greedy( self,time_allowance=60.0 ):
		cities = self._scenario.getCities()
		default_costs = np.array(self.createCostMatrix(cities))
		sorted_queue = self.createSortedCostList(cities)
		num_cities = len(cities)
		start_index = 0
		start_time = time.time()
		count = 0
		route = []
		while not sorted_queue.empty():
			path = sorted_queue.delete_min()
			cost = default_costs[path[0], path[1]]
			cityFrom = cities[path[0]]
			cityTo = cities[path[1]]
			if cost < float('inf'):
				indexTo = route.index(cityTo) if cityTo in route else -1
				indexFrom = route.index(cityFrom) if cityFrom in route else -1
				if indexFrom >= 0 and indexTo >= 0:
					continue
				if indexTo >= 0:
					route.insert(indexTo, cityFrom)
				elif indexFrom >= 0:
					route.insert(indexFrom + 1, cityTo)
				else:
					route.append(cityFrom)
					route.append(cityTo)
				default_costs[path[0], :] = float('inf')
				default_costs[:, path[1]] = float('inf')
		end_time = time.time()
		bssf = TSPSolution(route)
		results = {}
		results['cost'] = bssf.cost
		results['time'] = end_time - start_time
		results['count'] = 1
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results



	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints:
		max queue size, total number of states created, and number of pruned states.</returns>
	'''

	def branchAndBound( self, time_allowance=60.0 ):
		queue = HeapPriorityQueue()
		cities = self._scenario.getCities()
		costs = self.createCostMatrix(cities)
		paths = np.array([i for i in range(0, len(costs))])
		state = State(0, costs, 0, paths, paths, np.array([0]))
		states = {}
		states[0] = state
		start_time = time.time()
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
		

	def getInitialBSSF(self, time_allowance):
		results = self.defaultRandomTour(time_allowance)
		return results['cost']

	def createCostMatrix(self, cities):
		costs = []
		for i in range(len(cities)):
			costs.append([])
			for j in range(len(cities)):
				cost = cities[i].costTo(cities[j])
				costs[i].append(cost)

		return costs
	
	def createSortedCostList(self, cities):
		queue = HeapPriorityQueue()
		for i in range(len(cities)):
			for j in range(len(cities)):
				cost = cities[i].costTo(cities[j])
				if cost != float('inf'):
					queue.insert((i, j), cost)
		return queue

	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found during search, the
		best solution found.  You may use the other three field however you like.
		algorithm</returns>
	'''

	def fancy( self,time_allowance=60.0 ):
		pass

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

	def expand(self):
		states = []
		for i in range(len(self.out)):
			index = self.out[i]#._index
			if index == self.index:
				continue
			cost = np.ndarray.copy(self.cost)
			edge_cost = cost[self.index, index]
			cost[:, index] = float('inf')
			cost[self.index, :] = float('inf')
			out = np.delete(self.out, np.where(self.out == self.index))
			inp = np.delete(self.inp, np.where(self.inp == index))
			route = np.append(self.route, index)
			state = State(index, cost, self.lower_bound + edge_cost, out, inp, route)
			states.append(state)
		return states
	def is_complete(self):
		return len(self.out) <= 1
	def reduce(self, cost):
		cost = np.array(cost)
		total = 0
		for i in range(len(self.out)):
			index = self.out[i]#._index
			minv = min(cost[index, :])
			if minv == float('inf'):
				return cost, float('inf')
			cost[index, :] -= minv
			total += minv
		for i in range(len(self.inp)):
			index = self.inp[i]#._index
			minv = min(cost[:, index])
			if minv == float('inf'):
				return cost, float('inf')
			cost[:, index] -= minv
			total += minv

		return cost, total
