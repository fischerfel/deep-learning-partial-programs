import tensorflow as tf
print tf.__version__
#import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import cPickle as pickle

from datetime import datetime
from graphnnSiamese import graphnn
from exp_utils import *
from tensorflow.python import debug as tf_debug

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='', help='visible gpu device')
parser.add_argument('--fea_dim', type=int, default=7, help='feature dimension')
parser.add_argument('--embed_dim', type=int, default=64, help='embedding dimension')
parser.add_argument('--embed_depth', type=int, default=2, help='embedding network depth')
parser.add_argument('--output_dim', type=int, default=64, help='output layer dimension')
parser.add_argument('--iter_level', type=int, default=5, help='iteration times')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--epoch', type=int, default=100, help='epoch number')
parser.add_argument('--batch_size', type=int, default=5, help='batch size')
parser.add_argument('--load_path', type=str, default=None, help='path for model loading, "#LATEST#" for restoring from the latest checkpoint')
parser.add_argument('--save_path', type=str, default='./saved_model/graphnn-model', help='path for model saving')
parser.add_argument('--log_path', type=str, default=None, help='path for training log')
parser.add_argument('--add_data_time', type=int, default=0, help='how many added data will be included into training')




if __name__ == '__main__':
	args = parser.parse_args()
	args.dtype = tf.float32
	print("=================================")
	print(args)
	print("=================================")

	os.environ["CUDA_VISIBLE_DEVICES"]=args.device
	Dtype = args.dtype
	NODE_FEATURE_DIM = args.fea_dim
	EMBED_DIM = args.embed_dim
	EMBED_DEPTH = args.embed_depth
	OUTPUT_DIM = args.output_dim
	ITERATION_LEVEL = args.iter_level
	LEARNING_RATE = args.lr
	MAX_EPOCH = args.epoch
	BATCH_SIZE = args.batch_size
	LOAD_PATH = args.load_path
	SAVE_PATH = args.save_path
	LOG_PATH = args.log_path
	ADD_TIME = args.add_data_time

	SHOW_FREQ = 1
	TEST_FREQ = 5
	SAVE_FREQ = 5
	#DATA_FILE_NAME = '../acfgSSL_{}/'.format(NODE_FEATURE_DIM)
	DATA_FILE_NAME = '../apdgFDroid_{}/'.format(NODE_FEATURE_DIM)
        #SOFTWARE=('openssl-1.0.1f-', 'openssl-1.0.1u-')
        #SOFTWARE=('SignalAndroid-X-',)
        SOFTWARE = get_sw_name(DATA_FILE_NAME)
	#OPTIMIZATION=('-O0', '-O1','-O2','-O3')
	OPTIMIZATION=('-X',)
	#COMPILER=('armeb-linux', 'i586-linux', 'mips-linux')
        COMPILER=('source-android', 'application-android')
	VERSION=('v0',)

	FUNC_NAME_DICT = {}

	##  Processing input begin  ##
	F_NAME = get_f_name(DATA_FILE_NAME, SOFTWARE, COMPILER, OPTIMIZATION, VERSION)
	FUNC_NUM, FUNC_NAME_DICT = get_f_dict(F_NAME)

        #print F_NAME, FUNC_NAME_DICT, NODE_FEATURE_DIM

	Gs, classes = read_graph(F_NAME, FUNC_NAME_DICT, NODE_FEATURE_DIM)
	print "{} graphs, {} functions".format(len(Gs), len(classes))

        perm = np.random.permutation(len(classes))
        print ("perm in exp.py", len(perm))

	Gs_train, classes_train, Gs_test, classes_test, Gs_test2, classes_test2 = partition_data(Gs, classes, [0.8,0.1,0.1], perm)

	print "Train: {} graphs, {} functions".format(len(Gs_train), len(classes_train))
	print "Test: {} graphs, {} functions".format(len(Gs_test), len(classes_test))
	print "Test2: {} graphs, {} functions".format(len(Gs_test2), len(classes_test2))

	#Processing the pairs for validation and testing
	try:
		valid_ids = pickle.load(open('valid.ids'))
		valid_epoch = generate_epoch_pair(Gs_test, classes_test, BATCH_SIZE, load_id=valid_ids)
	except:
		valid_epoch, valid_ids = generate_epoch_pair(Gs_test, classes_test, BATCH_SIZE, output_id=True)
		pickle.dump(valid_ids, open('valid.ids', 'w'))
	try:
		test_ids = pickle.load(open('test.ids'))
		test_epoch = generate_epoch_pair(Gs_test2, classes_test2, BATCH_SIZE, load_id=test_ids)
	except:
		test_epoch, test_ids = generate_epoch_pair(Gs_test2, classes_test2, BATCH_SIZE, output_id=True)
		pickle.dump(test_ids, open('test.ids', 'w'))


	#Deal with the added data for retraining
	add_pairs = []
	for i in range(1, ADD_TIME+1):
		new_pairs = read_added_pair('./added_data/added_data{}'.format(i), NODE_FEATURE_DIM)
		add_pairs += new_pairs

	print "Added data: {} pairs".format(len(add_pairs))

	##  Processing input end  ##

	gnn = graphnn(
			N_x = NODE_FEATURE_DIM,
			Dtype = Dtype,
			N_embed = EMBED_DIM,
			depth_embed = EMBED_DEPTH,
			N_o = OUTPUT_DIM,
			ITERATION_LEVEL = ITERATION_LEVEL,
			lr = LEARNING_RATE
		)

	gnn.init(LOAD_PATH, LOG_PATH)
#	gnn.sess = tf_debug.LocalCLIDebugWrapperSession(gnn.sess)

        loss_valid = []
        loss_train = []

	auc_valid = []
	auc_v_valid = []
	auc_jacc_valid = []
	avg_prec_jacc_valid = []

	epochs = []
	#Training:

        epochs.append(0)
	l = get_loss_epoch(gnn, Gs_test, classes_test, BATCH_SIZE, load_data=valid_epoch, title='valid')
        loss_valid.append(l)
	gnn.say( "Testing model: validation loss = {0} @ {1}".format(l, datetime.now()) )
	auc0, auc_v_0 = get_auc_epoch(gnn, Gs_test, classes_test, BATCH_SIZE, load_data=valid_epoch, title='valid')
	#auc_jacc0, avg_prec0 = get_auc_jacc_epoch(gnn, Gs_test, classes_test, BATCH_SIZE, load_data=valid_epoch, title='valid')
        auc_valid.append(auc0)
	auc_v_valid.append(auc_v_0)
	#auc_jacc_valid.append(auc_jacc0)
	#avg_prec_jacc_valid.append(avg_prec0)
	gnn.say( "Testing model: validation auc = {0} @ {1}".format(auc0, datetime.now()) )
	gnn.say( "Testing model: validation auc_v = {0} @ {1}".format(auc_v_0, datetime.now()) )
	#gnn.say( "Testing model: validation auc_jacc = {0} @ {1}".format(auc_jacc0, datetime.now()) )
	#gnn.say( "Testing model: validation avg_prec = {0} @ {1}".format(avg_prec0, datetime.now()) )


	best_auc = 0
	for i in range(1, MAX_EPOCH+1):
		train_epoch(gnn, Gs_train, classes_train, BATCH_SIZE, add_pairs)

		if (i % SHOW_FREQ == 0):
			l = get_loss_epoch(gnn, Gs_train, classes_train, BATCH_SIZE, title='train')
			gnn.say( "EPOCH {3}/{0}, loss = {1} @ {2}".format(MAX_EPOCH, l, datetime.now(), i) )

		if (i % TEST_FREQ == 0):
                        epochs.append(i)
	                l = get_loss_epoch(gnn, Gs_test, classes_test, BATCH_SIZE, load_data=valid_epoch, title='valid')
                        loss_valid.append(l)
			gnn.say( "Testing model: validation loss = {0} @ {1}".format(l, datetime.now()) )
                        l_train = get_loss_epoch(gnn, Gs_train, classes_train, BATCH_SIZE, title='train')
                        loss_train.append(l_train)
			auc, auc_v = get_auc_epoch(gnn, Gs_test, classes_test, BATCH_SIZE, load_data=valid_epoch, title='valid')
			#auc_jacc, avg_prec = get_auc_jacc_epoch(gnn, Gs_test, classes_test, BATCH_SIZE, load_data=valid_epoch, title='valid')
                        auc_valid.append(auc)
			auc_v_valid.append(auc_v)
			#auc_jacc_valid.append(auc_jacc)
			#avg_prec_jacc_valid.append(avg_prec)
			gnn.say( "Testing model: validation auc = {0} @ {1}".format(auc, datetime.now()) )
			gnn.say( "Testing model: validation auc_v = {0} @ {1}".format(auc_v, datetime.now()) )
			#gnn.say( "Testing model: validation auc_jacc = {0} @ {1}".format(auc_jacc, datetime.now()) )
			#gnn.say( "Testing model: validation avg_prec_jacc = {0} @ {1}".format(avg_prec, datetime.now()) )


		if (i % SAVE_FREQ == 0):
			path = gnn.save(SAVE_PATH, i)
			gnn.say("Model saved in {}".format(path))


	test_auc = get_auc_epoch(gnn, Gs_test2, classes_test2, BATCH_SIZE, load_data=test_epoch, title='test')
	gnn.say( "Final auc on testing set: {}".format(test_auc) )
        np.savez("loss_valid.npz", loss=loss_valid, epoch=epochs)
        np.savez("loss_train.npz", loss=loss_train, epoch=epochs)
        np.savez("auc.npz", auc=auc_valid, epoch=epochs)
        np.savez("auc_v.npz", auc=auc_v_valid, epoch=epochs)
	#np.savez("auc_jacc.npz", auc=auc_jacc_valid, epoch=epochs)
	#np.savez("avg_prec_jacc.npz", prec=avg_prec_jacc_valid, epoch=epochs)
        save_embed_epoch(gnn, Gs_test2, classes_test2, BATCH_SIZE, load_data=test_epoch, title='test')
