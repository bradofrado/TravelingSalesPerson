from TSPSolver import *
from tabulate import tabulate
import sys, getopt

def readData(filename):
	split = lambda line:line.split(' ')
	data = None
	with open(filename, 'r') as f:
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
		datam.append(bssf['time'])
		datam.append(bssf['soln'].cost)
		datam.append(bssf['max'])
		datam.append(bssf['count'])
		datam.append(bssf['total'])
		datam.append(bssf['pruned'])

	return data

def getTable(data):
	headers = ['# Cities',
	 				'Seed',
					'Running time (sec.)',
					'Cost of best tour found (*=optimal)',
					'Max # of stored states at a given time',
					'# of BSSF updates',
					'Total # of states created',
					'Total # of states pruned']
	
	table = tabulate(data, headers=headers, tablefmt='fancy_grid')
	return table

def write(table, filename='table.txt'):
	with open(filename, 'w', encoding='utf-8') as f:
		f.write(table)


def getargs():
	inputfile = 'in.txt'
	outputfile = 'table.txt'
	opts, args = getopt.getopt(sys.argv[1:], 'hi:o:')
	for opt, arg in opts:
		if opt == '-h':
			print('table.py -i <inputfile> -o <outputfile>')
			sys.exit()
		elif opt == '-i':
			inputfile = arg
		elif opt == '-o':
			outputfile = arg

	return inputfile, outputfile

if __name__ == '__main__':
	inputfile, outputfile = getargs()
	SCALE = 1
	data_range = { 'x':[-1.5*SCALE,1.5*SCALE], \
								'y':[-SCALE,SCALE] }
	solver = TSPSolver()
	time_allowance = 60
	data = readData(inputfile)
	calculateData(data)
	table = getTable(data)
	write(table, outputfile)
	