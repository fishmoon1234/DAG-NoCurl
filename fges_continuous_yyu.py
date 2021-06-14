#!/usr/local/bin/python


import os
import pandas as pd
import pydot
import re
import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
import glog as log
import networkx as nx
import utils
import time
import argparse
import pickle
from IPython.display import SVG

from pycausal.pycausal import pycausal as pc
from pycausal import search as s


def parser_test_file(s,d):
	# parse file content
	lines = s.split('\n')
	A = np.zeros(shape=(d,d))
	for line in lines:
		if line and line[0] != 'd':
			edge_one = line.replace(";", "").replace(" ", "")
			edge = re.sub("[\(\[].*?[\)\]]", "", edge_one)
			edge = edge.split("->")
	        # check one direction or both
			if len(edge) == 2:
				if "both" in edge_one:
					A[int(eval(edge[0])), int(eval(edge[1]))] = 1
					A[int(eval(edge[1])), int(eval(edge[0]))] = 1
				else:
					A[int(eval(edge[0])), int(eval(edge[1]))] = 1
	return A

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_sample_size', type=int, default=1000,
							help='the number of samples of data')
	parser.add_argument('--data_variable_size', type=int, default=30,
							help='the number of variables in synthetic generated data')
	parser.add_argument('--graph_type', type=str, default='erdos-renyi',
							choices=['barabasi-albert','erdos-renyi'],
							help='the type of DAG graph by generation method')
	parser.add_argument('--graph_degree', type=int, default=2,
							help='the number of degree in generated DAG graph')
	parser.add_argument('--graph_sem_type', type=str, default='linear-gauss',
							choices=['linear-gauss','linear-exp','linear-gumbel'],
							help='the structure equation model (SEM) parameter type')
	parser.add_argument('--repeat', type=int, default= 100,
							help='the number of times to run experiments to get mean/std')
	args = parser.parse_args()

	return args

def fit_FGS(X, trueG, d, pc):
	X_df = pd.DataFrame(X)

	tetrad = s.tetradrunner()
	tetrad.run(algoId='fges', dfs=X_df, scoreId='sem-bic-score', dataType='continuous',
			   maxDegree=-1, faithfulnessAssumed=True, verbose=True)

	tetrad.getNodes()
	tetrad.getEdges()

	dot_str = pc.tetradGraphToDot(tetrad.getTetradGraph())
	print('learning_done')
	(graphs,) = pydot.graph_from_dot_data(dot_str)
	print(graphs, file=open('fges_raw.txt', 'w'))
	# graphs.write_png('fges-continuous.png')
	result = repr(graphs)
	print('splitting')
	lines = result.split("\n")
	all_edges = []
	pairs = []
	for line in lines:
		edge = line.replace(";", "").replace(" ", "").split("->")
		if len(edge) == 2:
			all_edges.append(edge[0])
			all_edges.append(edge[1])
			pairs.append(edge)

	unique_edges = set(all_edges)

	matrix = {origin: {dest: 0 for dest in all_edges} for origin in all_edges}
	for p in pairs:
		matrix[p[0]][p[1]] += 1

	# import pprint
	# print(matrix, file=open("FGES_result.txt","a"))
	#	pc.stop_vm()

	file = open('fges_raw.txt', "r+")
	dot_file = file.read()

	print('reading done' + dot_file)

	fgsG = (parser_test_file(dot_file, d))


	#	f = open('trueG', 'r')
	#	l = [[float(num) for num in line.split()] for line in f ]

	#	trueG=l
	for i in range(d):
		for j in range(d):
			if ((abs(trueG[i][j])) > 0.1):
				if (fgsG[i][j] > 0.1):
					fgsG[j][i] = 0.

	return fgsG

def main(args, pc):

	finalfile = open('fges_acc_time', 'w')

	n, d = args.data_sample_size, args.data_variable_size

	for trial_index in range(args.repeat):
		t =  time.time()
	#	data_dir = os.path.join(os.getcwd(), 'data', 'dataG')
	#	df = pd.read_table(data_dir, sep="\t")
		file_name = 'data/' + str(args.data_sample_size) + '_' + str(args.data_variable_size) + '_' \
						+ str(args.graph_type) + '_' + str(args.graph_degree) + '_' \
						+ str(args.graph_sem_type) + '_' + str(trial_index) + '.pkl'
		f = open(file_name, "rb")
		G, pkldata = pickle.load(f)
		trueG = nx.to_numpy_array(G)

	#	from pycausal.pycausal import pycausal as pc
	#	pc = pc()
	#	pc.start_vm()

	#	from pycausal import search as s
		X_df = pd.from_numpy(pkldata)
		fgsG = fit_FGS(X_df, trueG, d, pc)

		G = nx.DiGraph(np.array(trueG))
		G_est = nx.DiGraph(np.array(fgsG))

		fdr, tpr, fpr, shd, nnz = utils.count_accuracy(G, G_est)
		finalfile.write('Accuracy: fdr {}, tpr {}, fpr {}, shd {}, nnz {}, time {}'.format(
						 fdr, tpr, fpr, shd, nnz, time.time() - t))
		finalfile.write("\n")


if __name__ == '__main__':

	pc = pc()
	pc.start_vm()

	args = parse_args()
	main(args, pc)

	pc.stop_vm()




