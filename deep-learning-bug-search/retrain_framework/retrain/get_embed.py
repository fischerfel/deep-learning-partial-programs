import tensorflow as tf
print tf.__version__
#import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from graphnnSiamese import graphnn
from exp_utils import *
import os,csv
import argparse
import cPickle as pickle
import glob


parser = argparse.ArgumentParser()
#parser.add_argument('--device', type=str, default='', help='visible gpu device')
parser.add_argument('--fea_dim', type=int, default=112, help='feature dimension')
parser.add_argument('--embed_dim', type=int, default=64, help='embedding dimension')
parser.add_argument('--embed_depth', type=int, default=2, help='embedding network depth')
parser.add_argument('--output_dim', type=int, default=64, help='output layer dimension')
parser.add_argument('--iter_level', type=int, default=5, help='iteration times')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--epoch', type=int, default=1, help='epoch number')
parser.add_argument('--batch_size', type=int, default=5, help='batch size')
parser.add_argument('--load_path', type=str, default="saved_model/graphnn-model-100", help='path for model loading, "#LATEST#" for restoring from the latest checkpoint')
parser.add_argument('--save_path', type=str, default='./saved_model/graphnn-model', help='path for model saving')
parser.add_argument('--log_path', type=str, default=None, help='path for training log')
parser.add_argument('--add_data_time', type=int, default=0, help='how many added data will be included into training')
parser.add_argument('--retrain_id', type=int, default=0, help='The retraining id.')
parser.add_argument('--data_path', type=str, default=None, help='path for embedding data')
parser.add_argument('--data_save_path', type=str, default='results/', help='path for saving embeded data')
parser.add_argument('--type', type=str, default='', help='sample type')
parser.add_argument('--graphs', type=int, default=1, help='create graph embedding')
parser.add_argument('--only_labeled_graphs', type=int, default=0, help='create graph embedding only for labeled graphs')
parser.add_argument('--nodes', type=int, default=1, help='create node emnedding')


def get_subgraphs(embed, mask, node_num, out_node_deg=0):
    sub_graphs = []
    for i, e in enumerate(embed):
        if np.sum(mask[i]) >= out_node_deg and i < node_num:
            sub_graphs.append(e)

    return sub_graphs

def get_labeled_subgraphs(embed, mask, node_num, node_labels, out_node_deg=0):
    sub_graphs = []
    for i, e in enumerate(embed):
        if np.sum(mask[i]) >= out_node_deg and i in node_labels and i < node_num:
            sub_graphs.append(e)

    return sub_graphs
if __name__ == '__main__':
    args = parser.parse_args()
    args.dtype = tf.float32
    print("=================================")
    print(args)
    print("=================================")


    #os.environ["CUDA_VISIBLE_DEVICES"]=args.device
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
    DATA_PATH = args.data_path
    DATA_SAVE_PATH = args.data_save_path
    TYPE = args.type
    GRAPHS = args.graphs
    NODES = args.nodes
    ONLY_LABELED_GRAPHS = args.only_labeled_graphs

    SHOW_FREQ = 1
    TEST_FREQ = 5
    SAVE_FREQ = 5

    # create the saving directory if it is not existing
    directory = os.path.dirname(DATA_SAVE_PATH)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # create graph
    gnn = graphnn(
			N_x = NODE_FEATURE_DIM,
			Dtype = Dtype,
			N_embed = EMBED_DIM,
			depth_embed = EMBED_DEPTH,
			N_o = OUTPUT_DIM,
			ITERATION_LEVEL = ITERATION_LEVEL,
			lr = LEARNING_RATE
        )
    #load trained model if there is one
    gnn.init(LOAD_PATH, None)


    # load file names
    FUNC_NAME_DICT = {}
    F_NAME = glob.glob(DATA_PATH+"*")
    print F_NAME
    FUNC_NUM, FUNC_NAME_DICT = get_f_dict(F_NAME)

    Gs, classes = read_graph(F_NAME, FUNC_NAME_DICT, NODE_FEATURE_DIM)
    #print "{} graphs, {} functions".format(len(Gs), len(classes))

    if ONLY_LABELED_GRAPHS:
        labeled_graphs = []
        for i in range(len(Gs)):
            g = Gs[i]
            if 'tm' in TYPE and 'checkServerTrusted(' in g.name:
                labeled_graphs.append(g)
            if 'hnvor' in TYPE and 'verify(' in g.name:
                labeled_graphs.append(g)
        Gs = labeled_graphs

    maxN = 0
    # find max node number
    for i in range(len(Gs)): #FIXME read data in batches and define maxN by batch
        maxN = max(maxN, Gs[i].node_num)

    feature_dim = len(Gs[0].features[0])
    X_input = np.zeros((len(Gs), maxN, feature_dim)) #FIXME causes memory error
    node_mask = np.zeros((len(Gs), maxN, maxN))
    #y_input = np.zeros((M))# no pair, so that we dont need label

    names = []
    statement_sizes = []
    node_labels = []
    for i in range(len(Gs)):
        g = Gs[i]
        names.append(g.name)
        #statement_sizes.append(g.node_num)
        statement_sizes.append(1)
	node_labels.append(g.node_labels)
        for u in range(g.node_num):
            X_input[i, u, :] = np.array( g.features[u] )
            for v in g.succs[u]:
                node_mask[i, u, v] = 1

    # get graph embeddings
    embed_vec = gnn.get_embed(X_input, node_mask)
    vul_vec = embed_vec / (np.sqrt(np.sum(np.square(embed_vec), axis=1, keepdims=True)) + 1e-20)

    # get subgraphs embeddings
    vertices = gnn.get_v_embed(X_input, node_mask)
    vertices = vertices / (np.sqrt(np.sum(np.square(vertices), axis=2, keepdims=True)) + 1e-20)

    subgraphs = []
    subgraphs_size = []
    subgraphs_name = []
    for i, v in enumerate(vertices):
        s = get_labeled_subgraphs(v, node_mask[i], Gs[i].node_num, node_labels[i]) # [len(s), embed_dim]
        if len(s) > 0:
            s_pad = np.zeros((maxN, EMBED_DIM)) # [maxN, embed_dim]
            s_pad[:len(s)] = s
            subgraphs.append(s_pad)
            subgraphs_size.append(len(s))
            subgraphs_name.append(Gs[i].name)
	    print('name', Gs[i].name)
	    print('labels', node_labels[i])

    # save data to csv
    name_np = np.asarray(names)
    statement_size_np = np.asarray(statement_sizes)
    new = np.hstack((vul_vec,name_np.reshape(len(Gs),1),statement_size_np.reshape(len(Gs),1)))
    col_num = EMBED_DIM + 2
    if GRAPHS:
        np.savetxt(DATA_SAVE_PATH + str(EMBED_DIM) + '_graphs_output_' + TYPE + '.csv', new, delimiter='\t', fmt=["%s"]*col_num)

    # save vertices to csv
    name_np = np.asarray(subgraphs_name)
    statement_size_np = np.asarray(subgraphs_size)
    new = np.hstack((np.reshape(subgraphs, (len(subgraphs), maxN * EMBED_DIM)), name_np.reshape(len(subgraphs), 1),statement_size_np.reshape(len(subgraphs), 1)))
    col_num = EMBED_DIM * maxN + 2
    if NODES:
        np.savetxt(DATA_SAVE_PATH + str(EMBED_DIM) + '_vertices_output_' + TYPE + '.csv', new, delimiter='\t', fmt=["%s"]*col_num)

    #load data
    #embed_arr = np.genfromtxt(DATA_SAVE_PATH + str(OUTPUT_DIM)+'_output.csv', delimiter='\t',)[:,:-2]
    #size = np.genfromtxt(DATA_SAVE_PATH + str(OUTPUT_DIM)+'_output.csv', delimiter='\t',)[:,-1]
    #with open(DATA_SAVE_PATH + str(OUTPUT_DIM)+'_output.csv', 'rb') as f:
    #    reader = csv.reader(f, delimiter='\t')


