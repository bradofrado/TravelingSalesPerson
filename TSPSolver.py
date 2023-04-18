#!/usr/bin/python3
from BranchBound import BranchBound
from PriorityQueueImplementations import HeapPriorityQueue
from TSPClasses import *


class TSPSolver:
	def __init__(self, scenario=None):
		self._scenario = scenario

	def setupWithScenario(self, scenario):
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

	def defaultRandomTour(self, time_allowance=60.0):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time() - start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation(ncities)
			route = []
			# Now build the route using the random permutation
			for i in range(ncities):
				route.append(cities[perm[i]])
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
	def greedy(self, time_allowance=60.0, rnd = False):
		start_time = time.time()
		cities = self._scenario.getCities()
		count = 0
		foundTour = False
		# maybe starting from the first city does not find a solution, so try different start cities
		while not foundTour and count < len(cities) and time.time() - start_time < time_allowance:
			curr = cities[count]
			visited = [curr]
			unvisited = cities[:]
			unvisited.remove(curr)
			while unvisited:
				closest = self.random_neighbor(curr, unvisited) if rnd else self.nearest_neighbor(curr, unvisited)
				if closest == None:
					break
				visited.append(closest)
				unvisited.remove(closest)
				curr = closest
			bssf = TSPSolution(visited)
			if bssf.cost < float('inf') and len(visited) == len(cities):
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
	
	def random_neighbor(self, curr, cities):
		cost = float('inf')
		count = 0
		MAX_ITER = 100
		while cost == float('inf') and count < MAX_ITER:
			city = random.choice(cities)
			cost = curr.costTo(city)
			count += 1

		if count >= MAX_ITER:
			return None

		return city

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
	def branchAndBound(self, time_allowance=60.0):
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

		# if the final bssf equals the initial, then return 0
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

	def fancy(self, time_allowance=60.0):
		start_time = time.time()

		# Get the cities
		cities = self._scenario.getCities()
		num_cities = len(cities)

		# Define the genetic algorithm parameters
		POPULATION_SIZE = 50
		PARENTS_SIZE = 10
		MUTATION_RATE = 0.1
		ELITE_SIZE = 5  # number of best individuals to carry over to the next generation
		MAX_ITER_IF_NO_CHANGE = num_cities * 10  # if the bssf hasn't changed in 50 generations, stop

		# Create the initial population
		population = []
		# bssf = Individual(self.greedy(time_allowance)['soln'].route)
		# population.append(bssf)  # just to get an initial good solution
		bssf = None

		while len(population) != POPULATION_SIZE:
			potential_ind = Individual(self.greedy(time_allowance, True)['soln'].route)
			if potential_ind.fitness != np.inf and potential_ind not in population:
				population.append(potential_ind)
				if bssf is None or potential_ind.fitness < bssf.fitness:
					bssf = potential_ind

		count = 0
		number_gens = 0
		while count < MAX_ITER_IF_NO_CHANGE and time.time() - start_time < time_allowance:
			number_gens += 1
			count += 1
			# Create the next generation
			next_gen = []

			# Add the elite individuals to the next generation without any modifications
			elites = sorted(population)
			next_gen.extend(elites[:ELITE_SIZE])

			# tournament selection for the parents
			while len(next_gen) < PARENTS_SIZE:
				potential_par = self.tournament_selection(population, 5)
				if potential_par not in next_gen:
					next_gen.append(potential_par)

			# Add the rest of the individuals to the next generation
			for i in range(POPULATION_SIZE - len(next_gen)):
				# Create a child
				child = self.ERX(next_gen[i % PARENTS_SIZE].route, next_gen[(i + 1) % PARENTS_SIZE].route)
				route_copy = [x for x in child.route]

				# Mutate the child
				mutate_child = Individual(route_copy)
				mutate_child.mutate(MUTATION_RATE)

				max_mutations = 10
				i = 0
				while not self.check_valid(mutate_child, next_gen) and i < max_mutations:
					i += 1
					# redo the mutation
					mutate_child = Individual(route_copy)
					mutate_child.mutate(MUTATION_RATE)

				if self.check_valid(mutate_child, next_gen):
					next_gen.append(mutate_child)
					if mutate_child.fitness < bssf.fitness:
						bssf = mutate_child
						count = 0
				elif self.check_valid(child, next_gen):
					next_gen.append(child)
					if child.fitness < bssf.fitness:
						bssf = child
						count = 0
				else:
					# if the child is not valid, then add the parent
					next_gen.append(next_gen[i % PARENTS_SIZE])
			# Replace the current population with the next generation
			population = next_gen

		end_time = time.time()
		sol = TSPSolution(bssf.route)
		results = {}
		results['cost'] = sol.cost
		results['time'] = end_time - start_time
		results['count'] = number_gens
		results['soln'] = sol
		results['max'] = None
		results['total'] = None
		results['pruned'] = None

		return results

	def tournament_selection(self, population, tournament_size=5):
		tournament = random.sample(population, tournament_size)
		winner = tournament[0]
		for t in tournament[1:]:
			if t.fitness < winner.fitness:
				winner = t
		return winner

	def ERX(self, parent1, parent2):
		# implement the edge recombination crossover method for a directed TSP
		parents = []
		parents.extend(parent1)

		adj_dict = {key: [] for key in parents}
		for i in range(len(parent1)):
			adj_dict[parent1[i]].append(parent1[(i + 1) % len(parent1)])
			adj_dict[parent2[i]].append(parent2[(i + 1) % len(parent2)])

		# randomly select a starting node
		start = random.choice(parent1)

		# create the child
		child = [start]
		while True:
			# get the neighbors of the current node
			neighbors = adj_dict[child[-1]]
			# remove the neighbors that are already in the child
			neighbors = [n for n in neighbors if n not in child]
			# if there are no neighbors left, then choose a random node that is not in the child
			if len(neighbors) == 0:
				neighbors = [n for n in parent1 if n not in child]
			# choose the neighbor with the fewest neighbors
			neighbor = min(neighbors, key=lambda x: len(adj_dict[x]))

			if child[-1].costTo(neighbor) == np.inf:
				self.repair(child, neighbor)
			else:
				child.append(neighbor)

			# final check (and repair) to make sure the child is valid
			if len(child) == len(parent1):
				if child[-1].costTo(child[0]) == np.inf:
					self.repair(child, child.pop(0), True)
				break
		return Individual(child)

	def check_valid(self, ind, next_gen):
		if ind.fitness == np.inf or ind in next_gen:
			return False
		return True

	def repair(self, child, neighbor, final_test=False):
		# repair the child by swapping the neighbor with the city that has the lowest cost to the neighbor
		# alternative repair: insert the city instead of swap?

		# get the cost of the neighbor to all other cities
		neighbor_costs = [neighbor.costTo(c) for c in child]
		# get the index of the city with the lowest cost to the neighbor
		min_idx = neighbor_costs.index(min(neighbor_costs))
		if final_test:  # we can't use the first city in the child since that is what it originally was
			neighbor_costs[min_idx] = np.inf
			min_idx = neighbor_costs.index(min(neighbor_costs))

		if min_idx == 0 or child[min_idx - 1].costTo(neighbor) != np.inf:
			child.insert(min_idx, neighbor)
		else:
			# try swapping with the second minimum city
			neighbor_costs[min_idx] = np.inf
			min_idx = neighbor_costs.index(min(neighbor_costs))

			# recheck if the second minimum city is valid
			if min_idx == 0 or child[(min_idx - 1)].costTo(neighbor) != np.inf:
				child.insert(min_idx, neighbor)
			else:
				# give up -- the child is invalid and cannot be repaired
				child.append(neighbor)

class Individual:
	def __init__(self, route):
		self.route = route
		self.fitness = self.calculate_fitness()

	def calculate_fitness(self):
		total_distance = 0
		for i in range(len(self.route)):
			total_distance += self.route[i].costTo(self.route[(i + 1) % len(self.route)])
		return total_distance

	def mutate(self, MUTATION_RATE):
		# Mutate an individual
		# Swap two cities in the individual's route

		for i in range(len(self.route)):
			if random.random() < MUTATION_RATE:
				# Select two random cities to swap
				city1_idx = random.randint(0, len(self.route) - 1)
				city2_idx = random.randint(0, len(self.route) - 1)

				# Swap the cities
				self.route[city1_idx], self.route[city2_idx] = self.route[city2_idx], self.route[city1_idx]
		self.fitness = self.calculate_fitness()
		return self

	# def repair(self):
	# 	# perform a repair on the individual
	# 	problem_edges = []
	# 	for i in range(len(self.route)):
	# 		for j in range(i + 1, len(self.route)):
	# 			if self.route[i].costTo(self.route[j]) == np.inf:
	# 				problem_edges.append((i, j))
	# 	for edge in problem_edges:
	# 		# switch the edge with the shortest edge that does not cause a problem
	#
	# 		# get the shortest edge that does not cause a problem
	# 		shortest = np.inf
	# 		shortest_edge = None
	# 		for i in range(len(self.route)):
	# 			for j in range(i + 1, len(self.route)):
	# 				if self.route[i].costTo(self.route[j]) != np.inf:
	# 					if self.route[i].costTo(self.route[j]) < shortest:
	# 						shortest = self.route[i].costTo(self.route[j])
	# 						shortest_edge = (i, j)
	# 		# switch the edge
	# 		self.route[edge[0]], self.route[edge[1]] = self.route[shortest_edge[0]], self.route[shortest_edge[1]]
	# 	self.fitness = self.calculate_fitness()
	# 	return self

	def __eq__(self, other):
		return self.route == other.route

	def __lt__(self, other):
		return self.fitness < other.fitness

	def __gt__(self, other):
		return self.fitness > other.fitness


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
			index = self.out[i]  # ._index
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

			# When we create a new state, we reduce the cost matrix which is O(n^2) time
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
