from TSPSolver import *
from tabulate import tabulate
import sys, getopt

TIME = 60

def readData():
	split = lambda line:line.split(' ')
	data = None
	with open(sys.stdin.fileno()) as f:
		lines = f.readlines()
		data = [[int(char.replace('\n', '')) for char in split(line)] for line in lines]

	return data

def calculateData(data):
	for datam in data:
		npoints = datam[0]
		seed = datam[1]
		plist = newPoints(npoints, seed, data_range)
		scenario = Scenario(plist, "Hard (Deterministic)", seed)
		solver.setupWithScenario(scenario)
		bssf = solver.branchAndBound(time_allowance)
		print(bssf, file=sys.stderr)
		cost = bssf['cost']
		time = bssf['time']
		if (time < TIME):
			cost = '*' + str(cost)
		datam.append(time)
		datam.append(cost)
		datam.append(bssf['max'])
		datam.append(bssf['count'])
		datam.append(bssf['total'])
		datam.append(bssf['pruned'])

	return data

def getTable(data, format='fancy_grid'):
	headers = ['# Cities',
	 				'Seed',
					'Running time (sec.)',
					'Cost of best tour found',
					'Max # of stored states at a given time',
					'# of BSSF updates',
					'Total # of states created',
					'Total # of states pruned']
	table = tabulate(data, headers=headers, tablefmt=format, maxheadercolwidths=8)
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
	calculateData(data)
	table = getTable(data, format)
	write(table)
	