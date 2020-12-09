import os
import cPickle as pickle
import sys
import datetime
import multiprocessing
from raw_graphs import *
import tensorflow as tf
import numpy as np
from retrain.exp_utils import *
from retrain.graphnnSiamese import graphnn
import hashlib
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, default='OBJ_obj2txt', help='Target function name')
parser.add_argument('--thread', type=int, default=1, help='Thread number')
parser.add_argument('--sample_rate', type=float, default=0.1, help='probability for sampling')
parser.add_argument('--embed_dim', type=int, default=64, help='embedding dimension')
parser.add_argument('--embed_depth', type=int, default=2, help='embedding network depth')
parser.add_argument('--output_dim', type=int, default=64, help='output layer dimension')
parser.add_argument('--iter_level', type=int, default=5, help='iteration times')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--load_path', type=str, default=None, help='path for model loading, "#LATEST#" for restoring from the latest checkpoint')
parser.add_argument('--retrain_id', type=int, default=0, help='The retraining id.')



if __name__ == '__main__':
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
	SAMPLE_PROB = args.sample_rate

	##Get the embedding vector of OBJ_obj2txt
	vul_vec = np.load('output/vul_vec_retrain{}.npy'.format(args.retrain_id))
	print vul_vec.shape


	splits = args.thread

	inf = open('graphlist', 'r')

	#Get the files to process
	num = 0
	files = []
	for line in inf:
		##process line
		if np.random.uniform() > SAMPLE_PROB:
			continue
		files.append(line.strip())

	print len(files)




	def thread_job(tid, flist):
		print "Thread {} begins".format(tid)
		#Load the model
		DEVICE = '/gpu:{}'.format(tid % 8)
		gnn = graphnn(
				N_x = NODE_FEATURE_DIM,
				Dtype = Dtype, 
				N_embed = EMBED_DIM,
				depth_embed = EMBED_DEPTH,
				N_o = OUTPUT_DIM,
				ITERATION_LEVEL = ITERATION_LEVEL,
				lr = LEARNING_RATE,
				device = DEVICE,
				assigned_vec = vul_vec
			)
		gnn.init(LOAD_PATH, None)
		#gnn.init('./gnnmodel/gnnmodel','log',True)

		def turn_to_epoch(Glist, M, st):
			if (st + M > len(Glist)):
				M = len(Glist) - st
			ed = st + M

			maxN = 0
			for g_id in range(st, ed):
				raw_g = Glist[g_id].g
				maxN = max(maxN, len(raw_g.nodes()))

			X_in = np.zeros((M, maxN, 7))
			mask_in = np.zeros((M, maxN, maxN))
			name = []
			sha1s = []
			for g_id in range(st, ed):
				sha1 = ''
				name.append(Glist[g_id].funcname)
				raw_g = Glist[g_id].g
				for u in raw_g.nodes():
					feat = raw_g.node[u]['v']

					feat[0] = len(set(feat[0]))
					feat[1] = len(set(feat[1]))

					X_in[g_id-st, u] = np.array(feat[:6]+feat[7:])
					sha1 += str(feat)
					for v in raw_g.successors(u):
						sha1 += str(u)
						sha1 += str(v)
						mask_in[g_id-st, u, v] = 1

				sha1s.append(hashlib.sha1(sha1).hexdigest())
				#print sha1s

			return X_in, mask_in, name, sha1s
						


		#Begin processing
		def process_graph(Glist):
			BATCH_SIZE = 5

			ret = []
			st = 0
			while st < len(Glist):
				X_in, mask_in, name, sha1s = turn_to_epoch(Glist, BATCH_SIZE, st)
				cos_s = gnn.get_score(X_in, mask_in)
				for i in range(len(name)):
					ret.append( (name[i], cos_s[i], sha1s[i]) )
				st += BATCH_SIZE

			return ret


		num = 0
		tot_num = len(files)

		with open('output/retrain{}_thread{}.txt'.format(args.retrain_id, tid),'w') as ofile:
			for fname in flist:
				try:
					cur_raw_gs = pickle.load(open(fname, 'r'))
				except:
					continue

				embed = process_graph(cur_raw_gs.raw_graph_list)
				###embed: [(funcname, cos, sha1), (funcname, cos, sha1)]

				for elem in embed:
					if elem[1] > 0.9:
						ofile.write('{}@{} : {} : {}\n'.format(elem[0], fname, elem[1], elem[2]))
				ofile.flush()
				

				num += 1
				if num % 100 == 0:
					print "Thread {} : processed {} / {} @ {}".format(tid, num, len(flist), datetime.datetime.now())
					#if num >= 100:
					#	break


		print 'Thread {} is done.'.format(tid)




	tasks = []
	for i in range(splits):
		tasks.append([])
	for i in range(len(files)):
		tasks[i%splits].append(files[i])

	print len(tasks)
	for task in tasks:
		print len(task)

	threads = []
	for i in range(splits):
		t = multiprocessing.Process(target = thread_job, args = (i, tasks[i]))
		threads.append(t)
		t.start()

	for t in threads:
		t.join()

	# Merge
	vals = []
	for i in range(splits):
		inf = open('output/retrain{}_thread{}.txt'.format(args.retrain_id, i))
		for line in inf:
			info = line.strip().split(' : ')
			vals.append((info[0], float(info[1]), info[2]))

	vals = sorted(vals, key=lambda x:x[1], reverse=True)
	with open('output/retrain{}.txt'.format(args.retrain_id), 'w') as ofile:
		for elem in vals:
			ofile.write('{} : {} : {}\n'.format(elem[0], elem[1], elem[2]))
