from TSPSolver import *
from tabulate import tabulate
import sys, getopt

TIME = 600

def readData():
	split = lambda line:line.split(' ')
	data = None
	with open(sys.stdin.fileno()) as f:
		lines = f.readlines()
		data = [[int(char.replace('\n', '')) for char in split(line)] for line in lines]

	return data

def calculateData(data, algorithms):
	ret = []
	for i in range(len(data)):
		datam = data[i]
		npoints = datam[0]
		retd = [npoints]
		greedyCost = 0
		for j in range(len(algorithms)):
			costs = []
			times = []
			for seed in datam[1:]:
				plist = newPoints(npoints, seed, data_range)
				scenario = Scenario(plist, "Hard (Deterministic)", seed)
				solver.setupWithScenario(scenario)
				bssf = algorithms[j](time_allowance)
				print(bssf, file=sys.stderr)
				cost = bssf['cost']
				time = bssf['time']
				costs.append(cost)
				times.append(time)

			avg_cost = sum(costs) / len(costs)
			if avg_cost != float('inf'):
				avg_cost = round(avg_cost)
			avg_time = sum(times) / len(times)

			if avg_time < time_allowance or j > 2:
				retd.append(avg_cost)
				retd.append(avg_time)
			else:
				retd.append('TB')
				retd.append('TB')
				if j > 1:
					retd.append('TB')
				continue
			if j == 1:
				greedyCost = avg_cost
			if j > 0:
				if j == 1:
					lastCost = retd[1]
				else:
					lastCost = greedyCost
				percent = cost / lastCost
				retd.append(percent)

		ret.append(retd)

	return ret

def getTable(data, algorithms, format='fancy_grid'):
	#headers = ['Random', 'Greedy', 'Branch and Bound', 'Our Algorithm']
	subheader = ['# Cities']
	for i in range(len(algorithms)):
		subheader.extend(['Time (sec)', 'Path Length'])
		if i == 1:
			subheader.append('% of random')
		elif i > 1:
			subheader.append('% of greedy')
	table = tabulate(data, headers=subheader, tablefmt=format, maxheadercolwidths=8)
	return table

def write(table):
	with open(sys.stdout.fileno(), 'w', encoding='utf-8') as f:
		f.write(table)


def getargs():
	format = 'fancy_grid'
	opts, args = getopt.getopt(sys.argv[1:], 'hf:')
	for opt, arg in opts:
		if opt == '-h':
			print('table.py -f <table format>', file=sys.stderr)
			sys.exit()
		elif opt == '-f':
			format = arg

	return format

if __name__ == '__main__':
	format = getargs()
	SCALE = 1
	data_range = { 'x':[-1.5*SCALE,1.5*SCALE], \
								'y':[-SCALE,SCALE] }
	solver = TSPSolver()
	time_allowance = TIME
	data = readData()
	algorithms = [solver.fancy]
	data = calculateData(data, algorithms)
	table = getTable(data, algorithms, format)
	write(table)
	