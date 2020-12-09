import numpy as np
from raw_graphs import *
import cPickle as pickle
import os
import argparse

def output(g, ofile):
	ofile.write('{} {}||#added\n'.format(g.__len__(), g.funcname))
	for n in g.g.nodes():
		feat = g.g.node[n]['v']
		for i in [0,1,2,3,4,5,7]:
			if (i <= 1):
				ofile.write('{} '.format(len(feat[i])))
			else:
				ofile.write('{} '.format(feat[i]))
		ofile.write('{}'.format(len(g.g.succ[n])))
		for suc in g.g.succ[n]:
			ofile.write(' {}'.format(suc))
		ofile.write('\n')


parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, help='path for the source file to add')
parser.add_argument('--target', type=str, default='OBJ_obj2txt', help='Target function name')
parser.add_argument('--id', type=int, default=0, help='number of retraining')
parser.add_argument('--copies', type=int, default=50, help = 'how many copies to add into the training data')
parser.add_argument('--label', type=int, default=-1, help='-1 for negative cases, 1 for positive cases')

if __name__ == '__main__':
	args = parser.parse_args()
	if args.label != 1 and args.label != -1:
		raise Exception("Label must be 1 or -1")

	TARGET = args.target
	gfile = open('./retrain/added_data/added_data{}.graph'.format(args.id), 'w')
	idfile = open('./retrain/added_data/added_data{}.id'.format(args.id), 'w')
	inf = open(args.src, 'r')
	lines = inf.readlines()

	gfile.write('{}\n'.format(len(lines)+1))

	vul_gs = pickle.load(open('vul_data/openssl1.0.1a_x86.ida','r'))
	for g in vul_gs.raw_graph_list:
		if g.funcname == TARGET:
			print "hit"
			output(g, gfile)

	copies = args.copies
	cur_gid = 0
	for line in lines:
		cur_gid += 1
		funcinfo, val, _ = line.strip().split(' : ')
		print 'Adding {}'.format(funcinfo)
		funcname, filename = funcinfo.split('@')
		Gs = pickle.load(open(filename, 'r'))
		for g in Gs.raw_graph_list:
			if g.funcname == funcname:
				###   Process
				print "hit"
				output(g, gfile)
				for num in range(copies):
					idfile.write('0 {} {}\n'.format(cur_gid, args.label))



