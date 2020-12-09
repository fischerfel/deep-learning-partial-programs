import numpy as np
from raw_graphs import *
import cPickle as pickle
import os
import tensorflow as tf
from retrain.exp_utils import *
from retrain.graphnnSiamese import graphnn
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, default='OBJ_obj2txt', help='Target function name')
parser.add_argument('--embed_dim', type=int, default=64, help='embedding dimension')
parser.add_argument('--embed_depth', type=int, default=2, help='embedding network depth')
parser.add_argument('--output_dim', type=int, default=64, help='output layer dimension')
parser.add_argument('--iter_level', type=int, default=5, help='iteration times')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--load_path', type=str, default=None, help='path for model loading, "#LATEST#" for restoring from the latest checkpoint')
parser.add_argument('--src', type=str, help='path for the source file to add')


if __name__ == '__main__':
	#Load model
	Dtype = tf.float32
	args = parser.parse_args()
	TARGET = args.target
	NODE_FEATURE_DIM = 7
	EMBED_DIM = args.embed_dim
	EMBED_DEPTH = args.embed_depth
	ITERATION_LEVEL = args.iter_level
	OUTPUT_DIM = args.output_dim
	LEARNING_RATE = args.lr
	LOAD_PATH = args.load_path


	gnn = graphnn(
			N_x = NODE_FEATURE_DIM,
			Dtype = Dtype, 
			N_embed = EMBED_DIM,
			depth_embed = EMBED_DEPTH,
			N_o = OUTPUT_DIM,
			ITERATION_LEVEL = ITERATION_LEVEL,
			lr = LEARNING_RATE
		)
	gnn.init(LOAD_PATH, None)
	vul_gs = pickle.load(open('vul_data/openssl1.0.1a_x86.ida','r'))
	for g in vul_gs.raw_graph_list:
		if g.funcname == TARGET:
			node_num = len(g.g.nodes())
			X_in = np.zeros((1, node_num, 7))
			mask_in = np.zeros((1, node_num, node_num))
			for u in g.g.nodes():
				feat = g.g.node[u]['v']
				feat[0] = len(set(feat[0]))
				feat[1] = len(set(feat[1]))
				X_in[0, u] = np.array(feat[:6] + feat[7:])
				for v in g.g.successors(u):
					mask_in[0,u,v] = 1

			embed_vec, = gnn.get_embed(X_in, mask_in)
			vul_vec = embed_vec / (np.sqrt(np.sum(embed_vec ** 2)) + 1e-20)

	print vul_vec


	src_f = open(args.src, 'r')
	for line in src_f:
		funcinfo, val, _ = line.strip().split(' : ')
		print 'Testing {}'.format(funcinfo)
		funcname, filename = funcinfo.split('@')
		Gs = pickle.load(open(filename, 'r'))
		for g in Gs.raw_graph_list:
			if g.funcname == funcname:
				###   Process
				node_num = len(g.g.nodes())
				X_in = np.zeros((1, node_num, 7))
				mask_in = np.zeros((1, node_num, node_num))
				for u in g.g.nodes():
					feat = g.g.node[u]['v']
					feat[0] = len(set(feat[0]))
					feat[1] = len(set(feat[1]))
					X_in[0, u] = np.array(feat[:6] + feat[7:])
					for v in g.g.successors(u):
						mask_in[0,u,v] = 1

				embed_vec, = gnn.get_embed(X_in, mask_in)
				test_vec = embed_vec / (np.sqrt(np.sum(embed_vec ** 2)) + 1e-20)
				print np.sum(vul_vec * test_vec)
