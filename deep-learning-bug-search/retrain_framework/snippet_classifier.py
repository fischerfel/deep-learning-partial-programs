'''
Snippet classifier using pretrained embeddings

Author: Huang Xiao
Group: Cognitive Security Technologies
Institute: Fraunhofer AISEC
Mail: huang.xiao@aisec.fraunhofer.de
Copyright@2017
'''

import numpy as np
from mxnet import nd, autograd, gluon
import mxnet as mx
import itertools
import csv
import os
import sys
import pickle
import logging
import matplotlib.pyplot as plt
from termcolor import colored
import argparse
from argparse import ArgumentParser as argparser
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA, PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale, normalize
from sklearn.metrics import precision_score, recall_score, log_loss, precision_recall_curve, roc_curve, roc_auc_score, auc
from h3mlcore.utils import H3Plot
from bokeh.plotting import figure, output_file, show, save
from bokeh.layouts import gridplot


def duplicates(X, y, names):
    duplicates = set()
    for i, x1 in enumerate(X):
        duplicate_names = []
        for j, x2 in enumerate(X):
            if y[i] != y[j]:
                if np.array_equal(x1, x2):
                    duplicates.add(i)
                    duplicate_names.append(names[j])
                    print("duplicates: ", str(names[i]) + " " + str(names[j]))

        # if len(duplicate_names) != 0:
        #    print "-------------------------------------------------------"
        #    print "WARNING: " + str(names[i]) + "has ross-class duplicates"
        #    print str(duplicate_names)

    X_uniq = np.delete(X, list(duplicates), 0)
    y_uniq = np.delete(y, list(duplicates))
    assert(np.shape(X_uniq)[0] == np.shape(y_uniq)[0])
    print("Unique samples: ", np.shape(X_uniq)[0])

    return X_uniq, y_uniq


def metric_auc(labels, predicts):
    '''
    compute the auc for ROC curve for both binary/multi-class cases
    assume labels e.g., 0,1,2,3,...
    '''
    num_classes = np.unique(labels).size
    auc_scores = []
    labels = np.array(labels, dtype=int)
    # if num_classes > 2:
    for c in range(num_classes):
        binary_pred = []
        for p in predicts:
            binary_pred.append(p[c])
        # for each class
        fpr, tpr, thres = roc_curve(
            labels, np.array(binary_pred), pos_label=c)
        auc_scores.append(auc(fpr, tpr))
    # else:
    #     fpr, tpr, thres = roc_curve(labels, np.array(predicts[:, 1]))
    #     auc_scores.append(auc(fpr, tpr))

    return np.mean(np.array(auc_scores))


def get_roc_curves(labels, predicts):
    '''
    compute ROC curves for both binary/multi-class cases
    assume labels e.g., 0,1,2,3,...
    '''
    num_classes = np.unique(labels).size
    roc_curves = dict()
    labels = np.array(labels, dtype=int)
    # if num_classes > 2:
    for c in range(num_classes):
        binary_pred = []
        for p in predicts:
            binary_pred.append(p[c])
        # for each class
        fpr, tpr, thres = roc_curve(
            labels, np.array(binary_pred), pos_label=c)
        roc_curves[c] = (fpr, tpr, thres)
    # else:
    #     fpr, tpr, thres = roc_curve(labels, np.array(predicts[:, 1]))
    #     roc_curves[1] = (fpr, tpr, thres)
    return roc_curves


def metric_ce(labels, predicts):
    '''
    cross entropy for both binary/mulit-class cases
    assume labels e.g., 0,1,2,3,...
    '''
    ce = 0
    for l, p in zip(labels, predicts):
        ce += -np.log(p[int(l)])
    return ce / labels.shape[0]


def cal_pr(labels, predits, pos_label=1, classes=None, is_binary=False):
    if not is_binary:
        precision = precision_score(
            labels, predits, labels=classes, pos_label=pos_label, average=None)
        recall = recall_score(labels, predits, labels=classes,
                              pos_label=pos_label, average=None)
    else:
        precision = precision_score(
            labels, predits, labels=classes, pos_label=pos_label, average='binary')
        recall = recall_score(labels, predits, labels=classes,
                              pos_label=pos_label, average='binary')
    return precision, recall


def getDataIter(X, y,
                batch_size=10,
                split=0.4,
                is_binary=False):

    Xtr, Xtt, ytr, ytt = train_test_split(
        X, y, test_size=split, shuffle=True)

    # balance training set
    #label_dist_tr = [np.sum(ytr < 0), np.sum(ytr > 0)]
    #max_balance_size = min(label_dist_tr)
    #secure_idx_tr = np.argwhere(ytr < 0).ravel()
    #insecure_idx_tr = np.argwhere(ytr > 0).ravel()
    # if secure_idx_tr.size > insecure_idx_tr.size:
    #    resample_idx_tr = np.random.choice(
    #        secure_idx_tr, max_balance_size, replace=False)
    #    resample_idx_tr = np.r_[resample_idx_tr, insecure_idx_tr]
    # else:
    #    resample_idx_tr = np.random.choice(
    #        insecure_idx_tr, max_balance_size, replace=False)
    #    resample_idx_tr = np.r_[resample_idx_tr, secure_idx_tr]
    #Xtr_resampled = Xtr[resample_idx_tr]
    #ytr_resampled = ytr[resample_idx_tr]
    Xtr_resampled = Xtr
    ytr_resampled = ytr
    print("Training size: {:d} | Test size: {:d}".format(
        Xtr_resampled.shape[0], Xtt.shape[0]))
    output_labels = np.sort(np.unique(ytr_resampled))
    raw_dataset = {'train': Xtr_resampled,
                   'test': Xtt,
                   'train_labels': ytr_resampled,
                   'test_labels': ytt,
                   'labels': output_labels
                   }

    if is_binary:
        ytr_resampled_binary = np.array(ytr_resampled > 0, dtype=int)
        ytt_binary = np.array(ytt > 0, dtype=int)
        train_iter = mx.io.NDArrayIter(
            data=Xtr_resampled, label=ytr_resampled_binary, batch_size=batch_size, shuffle=True, label_name='softmax_label')
        test_iter = mx.io.NDArrayIter(
            data=Xtt, label=ytt_binary, batch_size=batch_size, shuffle=True, label_name='softmax_label')
    else:
        ytr_resampled = np.array(
            [np.where(output_labels == l)[0][0] for l in ytr_resampled])
        ytt = np.array([np.where(output_labels == l)[0][0] for l in ytt])
        train_iter = mx.io.NDArrayIter(
            data=Xtr_resampled, label=ytr_resampled, batch_size=batch_size, shuffle=True, label_name='softmax_label')
        test_iter = mx.io.NDArrayIter(
            data=Xtt, label=ytt, batch_size=batch_size, shuffle=True, label_name='softmax_label')

    # TODO: write out a test CSV for prediction -> to be removed
    if not os.path.exists('datasets/test.csv'):
        with open('datasets/test.csv', 'w') as fout:
            csv_writer = csv.writer(fout, delimiter='\t')
            csv_writer.writerows(Xtt)

    return train_iter, test_iter, raw_dataset


def NodeEmbedInCSV(path='datasets/in_out_neighbours/',
                   types=None,
                   embed_size=64,
                   sep='\t',
                   on_types_only=False):
    '''
    get iterator from dataset with node embeddings
    '''

    prefix = ''.join([path, str(embed_size), '_vertices_output_'])
    csv_file_list = []
    labels_all = []
    names_all = []
    nodes_all = np.empty([0, embed_size])
    for label, t in itertools.product(['insecure', 'secure'], types):
        csv_file = prefix + label + '_' + t + '.csv'
        if not os.path.exists(csv_file):
            logging.info("{:s} does not exist, skip...".format(csv_file))
            continue
        csv_file_list.append(csv_file)
        if on_types_only:
            # y labels contain only insecure types from 0 to M
            y = types.index(t)
        else:
            # y labels contains types and separate secure/insecure e.g., -2, -1, 1, 2
            y = -(types.index(t) + 1) if label == 'secure' else types.index(t) + 1
        with open(csv_file, 'rb') as fd:
            reader = csv.reader(fd, delimiter=sep)
            nodes = np.empty([0, embed_size], dtype=np.float)
            labels = []
            names = []
            for row in reader:
                graph_name = row[len(row) - 2]
                node_num = int(row[len(row) - 1])
                v = np.array(row[:embed_size * node_num], dtype=np.float)
                v = np.reshape(v, (node_num, embed_size))
                nodes = np.vstack((nodes, v))
                labels.extend([y for _ in range(node_num)])
                names.extend([graph_name for _ in range(node_num)])
        logging.debug("{:d} node embeddings in csv file {:s}".format(
            len(labels), csv_file))
        nodes_all = np.vstack((nodes_all, nodes))
        labels_all.extend(labels)
        names_all.extend(names)
    # all files done
    nodes_all = normalize(nodes_all)
    labels_all = np.array(labels_all)
    print ">>> Done! {:d} embeddings | counts of labels {:s}".format(
        len(labels_all), str(np.bincount(labels_all + np.max(labels_all))))
    return nodes_all, labels_all, names_all


def readEmbeddings(csv_file, embed_size, sep):
    '''
    get embeddings from a csv file
    '''

    with open(csv_file, 'rb') as fd:
        reader = csv.reader(fd, delimiter=sep)
        nodes = np.empty([0, embed_size], dtype=np.float)
        names = []
        for row in reader:
            graph_name = row[len(row) - 2]
            node_num = int(row[len(row) - 1])
            v = np.array(row[:embed_size * node_num], dtype=np.float)
            v = np.reshape(v, (node_num, embed_size))
            nodes = np.vstack((nodes, v))
            names.extend([graph_name for _ in range(node_num)])

    return nodes, names

    #nodes = np.empty([0, embed_size], dtype=np.float)
    #labels = []
    # with open(csv_file, 'rb') as fd:
    #    reader = csv.reader(fd, delimiter=sep)
    #    for row in reader:
    #        if len(row) < embed_size:
    #            print colored(
    #                'Input data has a different embedding size, exit!', 'red')
    #            sys.exit()
    #        else:
    #            nodes = np.append(nodes, np.array(
    #                row[:embed_size], dtype=float).reshape(1, embed_size), axis=0)
    #            if len(row) > embed_size:
    #                labels.append(row[embed_size])
    # return nodes, np.array(labels)


def train(train_iter,
          val_iter=None,
          labels=[0, 1],
          epochs=10,
          dropout=0.2,
          show_cg=False,
          is_binary=False,
          save_freq=20,
          checkpoint='./checkpoints/',
          save_prefix='snp_classifier'):
    '''
    build the computation graph for training NN
    '''

    save_path_prefix = ''.join([checkpoint, save_prefix])

    if is_binary:
        labels = [0, 1]
    num_classes = len(labels)

    input_dims = train_iter.provide_data[0][1]
    # define network
    net = mx.sym.Variable('data')
    net = mx.sym.FullyConnected(net, name='fc1', num_hidden=64)
    net = mx.sym.Activation(net, name='relu1', act_type='relu')
    net = mx.sym.Dropout(net, name='dropout1', p=dropout)
    # net = mx.sym.max(net, axis=1)
    net = mx.sym.FullyConnected(net, name='fc2', num_hidden=32)
    net = mx.sym.Activation(net, name='relu2', act_type='relu')
    net = mx.sym.Dropout(net, name='dropout2', p=dropout)
    net = mx.sym.FullyConnected(net, name='fc3', num_hidden=num_classes)
    net = mx.sym.SoftmaxOutput(net, name='softmax')
    # visualize network
    if show_cg:
        graph = mx.viz.plot_network(
            symbol=net, shape={'data': input_dims, 'softmax_label': (input_dims[0], )})
        graph.view()
    # TODO: define callbacks
    # define module
    mod = mx.mod.Module(symbol=net,
                        context=[mx.cpu(0), mx.cpu(1), mx.cpu(2), mx.cpu(3)],
                        data_names=['data'],
                        label_names=['softmax_label'])

    # bind train data
    mod.bind(data_shapes=train_iter.provide_data,
             label_shapes=train_iter.provide_label)
    mod.init_params(initializer=mx.init.Xavier())
    lrs = mx.lr_scheduler.FactorScheduler(200, factor=0.9, stop_factor_lr=1e-3)
    mod.init_optimizer(
        optimizer='sgd', optimizer_params=(('learning_rate', 0.5), ('lr_scheduler', lrs), ))
    train_metric = mx.metric.create([mx.metric.CrossEntropy(), 'acc'])
    train_ce = []
    train_acc = []
    test_ce = []
    test_acc = []
    test_prec = []
    test_recall = []
    test_auc = []
    test_roc_curves = []
    test_metric = [mx.metric.Accuracy(), mx.metric.np(metric_ce),
                   mx.metric.np(metric_auc)]
    test_compose_metrics = mx.metric.CompositeEvalMetric()
    for met in test_metric:
        test_compose_metrics.add(met)

    for epoch in range(epochs):
        train_iter.reset()
        val_iter.reset()
        train_metric.reset()
        test_compose_metrics.reset()
        for batch in train_iter:
            mod.forward(batch, is_train=True)
            mod.update_metric(train_metric, batch.label)
            mod.backward()
            mod.update()
        train_ce.append(train_metric.get()[1][0])
        train_acc.append(train_metric.get()[1][1])

        # validate epoch on test set
        val_labels = []
        val_pred_proba = []
        val_pred = []
        for pred, i_batch, batch in mod.iter_predict(val_iter):
            if i_batch == 0:
                val_pred_proba = pred[0]
                val_labels = batch.label[0][:pred[0].shape[0]]
            else:
                val_pred_proba = mx.nd.concat(val_pred_proba, pred[0], dim=0)
                val_labels = mx.nd.concat(
                    val_labels, batch.label[0][:pred[0].shape[0]], dim=0)

        pred = np.argmax(val_pred_proba.asnumpy(), axis=1)
        test_compose_metrics.update([val_labels], [val_pred_proba])
        prec, recall = cal_pr(val_labels.asnumpy(), pred, pos_label=1,
                              classes=labels, is_binary=is_binary)
        test_acc.append(test_compose_metrics.get_metric(0).get()[1])
        test_ce.append(test_compose_metrics.get_metric(1).get()[1])
        test_auc.append(test_compose_metrics.get_metric(2).get()[1])
        test_prec.append(prec.tolist())
        test_recall.append(recall.tolist())
        test_roc_curves.append(get_roc_curves(
            val_labels.asnumpy(), val_pred_proba.asnumpy()))
        import time
        import datetime
        ts = time.time()
        dt = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        print dt
        print 'Epoch {:d}, {:s}: {:.2f} (train) / {:.2f} (test),  {:s}: {:.2f} (train) / {:.2f} (test),  {:s}: {:.2f} (test)'.format(
            epoch, train_metric.get()[0][0], train_ce[-1], test_ce[-1], train_metric.get()[0][1], train_acc[-1], test_acc[-1], 'Test AUC', test_auc[-1])
        if (epoch + 1) % save_freq == 0:
            mod.save_checkpoint(prefix=save_path_prefix, epoch=epoch + 1)

    # store evaluation results in a dict
    eval_results = {'train_ce': train_ce,
                    'test_ce': test_ce,
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                    'test_prec': test_prec,
                    'test_recall': test_recall,
                    'test_auc': test_auc,
                    'test_roc_curves': test_roc_curves}

    return mod, eval_results


def predict(mod, test_iter, labels=None):
    '''
    mod: mxnet module
    test_iter: Test DataIterator
    '''

    predict_logits = mod.predict(test_iter).asnumpy()
    if labels:
        predict_labels = [labels[np.argmax(predict_logits[r])] for r in range(
            predict_logits.shape[0])]
    else:
        predict_labels = [np.argmax(predict_logits[r]) for r in range(
            predict_logits.shape[0])]
    return np.array(predict_labels)


# ------ main -------

if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger('default')
    vis = H3Plot.DataViz(
        config={'dot_size': 2, 'colormap': 'Viridis', 'line_width': 1})

    DESCRIPTION = """
    --------------------------------------------------------------------------------------------------------
        Train or classify a given CSV dataset, using a simple mxnet MLP model.
        Data format:
            - The input CSV file for prediction should contain N rows of embeddings with
            a fixed embedding size set by embed_size, optionally can contain labels at
            the last column. The columns are separated by a tab.

        If you don't use '--predict' option, by default it will train on datasets set by
        the '--path', where you should include a list of CSV files as training data.

        After training, a model will be cached, which can be used for prediction. If you
        don't specify a checkpoint file explicitely, it will automatically determin the
        latest checkpoint to predict. In prediction, both labels and new embeddings will be
        written out to a file set by option '--output'.

        Note that the option '--types' fix the orders of input labels, that means, if you
        input '--types cipher,tls,hash', then the labels correspond as follows:
        {'secure_hash' : 0,
         'secure_tls' : 1,
         'secure_cipher': 2,
         'insecure_cipher': 3,
         'insecure_tls': 4,
         'insecure_hash': 5,}
        Since the labels info is not cached after training, you should be careful about
        using multi-class mode.

        For example:
        This will train a model,
        python snippet_classifier.py --path datasets/in_out_neighbours/ --epochs 100 --batch_size 50 --is_binary --should_plot

        This will predict on test.csv using latest model,
        python snippet_classifier.py --predict datasets/test.csv --should_plot
    --------------------------------------------------------------------------------------------------------
    """

    parser = argparser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=DESCRIPTION)
    parser.add_argument('--path', type=str, default='./datasets/in_out_neighbours/',
                        help='Root path to the datasets')
    parser.add_argument('-t', '--types', type=str, default='',
                        help='insecure embedding types, separated by comma, e.g, cipher,tls,hash')
    parser.add_argument('--embed_size', type=int, default=64,
                        help='embedding dimension size')
    parser.add_argument('--train_test_split', type=float,
                        default=0.4, help='train test split ratio')
    parser.add_argument('--is_binary', action='store_true',
                        help='use binary classification, default to multiclass')
    parser.add_argument('--on_types_only', action='store_true',
                        help='train/predict only on types, no secure/insecure info, default to false; this secure/insecure info, default to false; this option cancels *is_binary* option')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='batch size, default 50')
    parser.add_argument('--epochs', type=int, default=10,
                        help='epoch size, default 10')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Probability of dropout')
    parser.add_argument('--save_freq', type=int, default=20,
                        help='save checkpoint frequency, default 20 epochs')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/',
                        help='path to checkpoints, default ./checkpoints/')
    parser.add_argument('--save_prefix', type=str, default='snp_classifier',
                        help='prefix to the saved checkpoint, default snp_classifier')
    parser.add_argument('--save_npz', type=str, default='./plots/',
                        help='path the save the npz of the plots, default saved in .plots/')
    parser.add_argument('--should_plot', action='store_true',
                        help='should plot results, default false')
    parser.add_argument('--plot_method', type=str, default='pca',
                        help='choose a method to plot, can be pca or tsne')
    parser.add_argument('--predict', type=str, default=None,
                        help='predict embeddings and labels on certain input CSV files.')
    parser.add_argument(
        '--output', type=str, help='Output file to save predictions: embeddings and labels.')
    parser.add_argument('--model', type=str, default='LATEST',
                        help='which checkpoint to be used for prediction or generating embeddings. default to latest')

    args = parser.parse_args()
    if not args.types:
        INSECURE_TYPES = ['cipher', 'tls']
    else:
        INSECURE_TYPES = [elem for elem in args.types.split(',')]
    EMBED_SIZE = args.embed_size
    IS_BINARY = args.is_binary
    ON_TYPES_ONLY = args.on_types_only
    if ON_TYPES_ONLY:
        # force is_binary option to false, if we train only on types
        IS_BINARY = False
    TRAIN_TEST_SPLIT = args.train_test_split
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    DROPOUT = args.dropout
    ROOT_PATH = args.path
    SAVE_FREQ = args.save_freq
    SAVE_PREFIX = args.save_prefix
    CHECKPOINT_PATH = args.checkpoint
    SHOULD_PLOT = args.should_plot
    PLOT_METHOD = args.plot_method
    if PLOT_METHOD not in ['pca', 'tsne']:
        print colored(
            'Plot method is not supported, please only choose pca or tsne. exit!', 'red')
        sys.exit()
    PREDICT_INPUT = args.predict
    OUTPUT_FILE = args.output
    MODEL_PATH = args.model
    SAVE_NPZ = args.save_npz
    if not os.path.isdir(SAVE_NPZ):
        # create folder if does not exist
        os.mkdir(SAVE_NPZ)

    # load model first
    if MODEL_PATH == 'LATEST':
        latest_mtime = 0
        latest_chk = ''
        for fname in os.listdir(CHECKPOINT_PATH):
            fname_path = os.path.join(CHECKPOINT_PATH, fname)
            fstat = os.lstat(fname_path)
            if fstat.st_mtime > latest_mtime and fname.split('.')[-1] == 'params':
                latest_chk = fname_path
                latest_mtime = fstat.st_mtime
        latest_symbol = latest_chk.split('-')[0] + '-symbol.json'
    else:
        latest_chk = MODEL_PATH
        latest_symbol = latest_chk.split('-')[0] + '-symbol.json'
        print latest_chk
        print latest_symbol
    #if not os.path.exists(latest_symbol):
    #    print colored('Model symbol file ' + latest_symbol + ' does not exist. exit!', 'red')
    #    sys.exit()
    #else:
    #    print colored('Model: {:s} is loaded!'.format(latest_chk), 'yellow')
    #    load_epochs = int(latest_chk.split('-')[-1].split('.')[0])
    #    prefix = latest_chk.split('-')[0]
    #    mod = mx.mod.Module.load(prefix, load_epochs)
    if os.path.exists(latest_symbol):
        print colored('Model: {:s} is loaded!'.format(latest_chk), 'yellow')
        load_epochs = int(latest_chk.split('-')[-1].split('.')[0])
        prefix = latest_chk.split('-')[0]
        mod = mx.mod.Module.load(prefix, load_epochs)

    if PREDICT_INPUT:
        if not os.path.exists(PREDICT_INPUT):
            print colored(
                'Input CSV to predict does not exist. check your input!', 'red')
            sys.exit()
        embeddings_from_csv, names = readEmbeddings(
            PREDICT_INPUT, embed_size=EMBED_SIZE, sep='\t')
        data_iter = mx.io.NDArrayIter(
            embeddings_from_csv, batch_size=1)
        mod.bind(for_training=False, data_shapes=data_iter.provide_data)
        arg_params, aux_params = mod.get_params()
        predicted = predict(mod, data_iter)
        # done with predict, output last layer
        data_iter.reset()
        all_layers = mod.symbol.get_internals()
        new_embed_layer = all_layers['fc2_output']
        mod2 = mx.mod.Module(symbol=new_embed_layer,
                             label_names=None, context=mx.cpu())
        mod2.bind(for_training=False, data_shapes=data_iter.provide_data)
        mod2.set_params(arg_params, aux_params,
                        allow_missing=True, allow_extra=True)
        new_embeddings = np.empty((0, 32), dtype=float)
        for batch in data_iter:
            mod2.forward(batch, is_train=False)
            out_embed = mod2.get_outputs()[0].asnumpy()
            new_embeddings = np.append(new_embeddings, out_embed, axis=0)

        if not OUTPUT_FILE or not os.path.exists(OUTPUT_FILE):
            OUTPUT_FILE = ''.join(PREDICT_INPUT.split('.')[
                :-1]) + '_output.csv'

        with open(OUTPUT_FILE, 'w') as fout:
            csv_writer = csv.writer(fout, delimiter='\t')
            for i, l in enumerate(predicted):
                #csv_writer.writerow(np.append(new_embeddings[r], l))
                csv_writer.writerow([names[i], l])
            print 'Done. predicted embeddings and labels are dumped in: ', OUTPUT_FILE

        if SHOULD_PLOT:
            vis.output_file = 'predicts.html'
            # TODO: set labelnames
            p = vis.project2d(X=new_embeddings,
                              y=predicted,
                              method=PLOT_METHOD,
                              title='Embeddings after Training')
            #show(p)
    else:
        # train on datasets
        if ON_TYPES_ONLY:
            if len(INSECURE_TYPES) < 2:
                print(colored('Trying to train a multi-class classifier on one type! exit.', 'red'))
                sys.exit()
            else:
                print(colored("--- Training multi-class classifier only on types [{:s}] ---".format(', '.join(INSECURE_TYPES)), "yellow"))
        elif IS_BINARY:
            print(colored(
                "--- Training a binary classifier on secure/insecure embeddings ---", "yellow"))
        else:
            print(colored(
                "--- Training a multi-class classifier on secure/insecure embeddings with types ---", "yellow"))

        Xtr, ytr, names = NodeEmbedInCSV(path=ROOT_PATH,
                                         types=INSECURE_TYPES,
                                         embed_size=EMBED_SIZE,
                                         sep='\t',
                                         on_types_only=ON_TYPES_ONLY)
        Xtr, ytr = duplicates(Xtr, ytr, names)
        tr_iter, tt_iter, dataset = getDataIter(Xtr,
                                                ytr,
                                                split=TRAIN_TEST_SPLIT,
                                                batch_size=BATCH_SIZE,
                                                is_binary=IS_BINARY)
        mod, res = train(tr_iter,
                         tt_iter,
                         labels=dataset['labels'],
                         epochs=EPOCHS,
                         dropout=DROPOUT,
                         is_binary=IS_BINARY,
                         save_freq=SAVE_FREQ,
                         checkpoint=CHECKPOINT_PATH,
                         save_prefix=SAVE_PREFIX)

        if IS_BINARY:
            # test precision/recall per class
            # this is only valid for non-binary and non types only case
            testX = dataset['test']
            testy = dataset['test_labels']
            test_prcurves = []
            for idx, t in enumerate(INSECURE_TYPES):
                pos_l = idx + 1
                neg_l = - idx - 1
                test_idx = np.r_[np.where(testy == pos_l)[0],
                                 np.where(testy == neg_l)[0]]
                test_iter = mx.io.NDArrayIter(data=testX[test_idx],
                                              batch_size=BATCH_SIZE,
                                              shuffle=False,
                                              label_name='softmax_label')
                test_labels = testy[test_idx]
                test_labels = np.array(test_labels > 0, dtype=int)
                pred_prob = mod.predict(test_iter).asnumpy()
                prec, rec, thres = precision_recall_curve(
                    test_labels, pred_prob[:, 1])
                # calculate auc
                pr_auc = auc(rec, prec)
                test_prcurves.append((prec, rec, pr_auc))

        if SHOULD_PLOT:
            if not IS_BINARY:
                print len(res['test_prec'])
                num_classes = len(res['test_prec'][0])
                if ON_TYPES_ONLY:
                    label_names = INSECURE_TYPES
                else:
                    label_names = ['sec_' + INSECURE_TYPES[-l - 1]
                                   if l < 0 else 'insec_' + INSECURE_TYPES[l - 1] for l in dataset['labels']]
            else:
                num_classes = 2
                label_names = ['secure', 'insecure']
                dataset['train_labels'] = np.array(
                    dataset['train_labels'] > 0, dtype=int)

            p1, x_proj = vis.project2d(X=dataset['train'],
                                       y=dataset['train_labels'],
                                       method=PLOT_METHOD,
                                       title='Samples in 2D',
                                       legend=label_names)
            with open(SAVE_NPZ + 'train_samples_2d.npz', 'w') as fd:
                np.savez(fd, x_proj=x_proj,
                         labels=dataset['train_labels'], legend=label_names)

            # done with training, output last layer for the train set to visualize
            data_iter = mx.io.NDArrayIter(dataset['train'], batch_size=1)
            arg_params, aux_params = mod.get_params()
            all_layers = mod.symbol.get_internals()
            new_embed_layer = all_layers['fc2_output']
            mod2 = mx.mod.Module(symbol=new_embed_layer,
                                 label_names=None, context=mx.cpu())
            mod2.bind(for_training=False, data_shapes=data_iter.provide_data)
            mod2.set_params(arg_params, aux_params,
                            allow_missing=True, allow_extra=True)
            new_embeddings = np.empty((0, 32), dtype=float)
            for batch in data_iter:
                mod2.forward(batch, is_train=False)
                out_embed = mod2.get_outputs()[0].asnumpy()
                new_embeddings = np.append(new_embeddings, out_embed, axis=0)
            p2, x_proj_new = vis.project2d(X=new_embeddings,
                                           y=dataset['train_labels'],
                                           method=PLOT_METHOD,
                                           title='Embeddings after Training',
                                           legend=label_names)
            with open(SAVE_NPZ + 'train_embeddings_new2d.npz', 'w') as fd:
                np.savez(fd, x_proj_new=x_proj_new,
                         labels=dataset['train_labels'], legend=label_names)

            # plot train/test curves
            p3 = vis.simple_curves(range(EPOCHS),
                                   yvalues=[res['train_ce'], res['test_ce'],
                                            res['train_acc'], res['test_acc']],
                                   legend=['Train Cross - Entropy',
                                           'Test Cross - Entropy',
                                           'Train Accuracy',
                                           'Test Accuracy'],
                                   legend_loc='top_right',
                                   title='Train/test results',
                                   xlabel='epochs')
            with open(SAVE_NPZ + 'train_test_curves.npz', 'w') as fd:
                np.savez(fd,
                         epochs=range(EPOCHS),
                         yvalues=[res['train_ce'], res['test_ce'],
                                  res['train_acc'], res['test_acc']],
                         legend=['Train Cross - Entropy',
                                 'Test Cross - Entropy',
                                 'Train Accuracy',
                                 'Test Accuracy'])

            # plot test prec/recall per class curves
            test_prec_per_class = []
            test_recall_per_class = []
            for c in range(num_classes):
                test_prec_per_class.append(
                    [prec[c] if type(prec) == list else prec for prec in res['test_prec']])
                test_recall_per_class.append(
                    [recall[c] if type(recall) == list else recall for recall in res['test_recall']])
            p4 = vis.simple_curves(range(EPOCHS),
                                   yvalues=test_prec_per_class,
                                   legend=label_names,
                                   legend_loc='bottom_right',
                                   title='Test Precision',
                                   xlabel='epochs')
            with open(SAVE_NPZ + 'test_precision_per_class.npz', 'w') as fd:
                np.savez(fd,
                         epochs=range(EPOCHS),
                         yvalues=test_prec_per_class,
                         legend=label_names)

            p5 = vis.simple_curves(range(EPOCHS),
                                   yvalues=test_recall_per_class,
                                   legend=label_names,
                                   legend_loc='bottom_right',
                                   title='Test Recall',
                                   xlabel='epochs')
            with open(SAVE_NPZ + 'test_recall_per_class.npz', 'w') as fd:
                np.savez(fd,
                         epochs=range(EPOCHS),
                         yvalues=test_recall_per_class,
                         legend=label_names)

            per_class_prcurves_plots = []
            if IS_BINARY:
                # PRCurve for binary case wrt. each class
                for i, curves in enumerate(test_prcurves):
                    # precision-recall curve plot
                    p = vis.simple_curves(curves[1],
                                          curves[0],
                                          xlabel='recall (dropout={:.2f})'.format(
                                              DROPOUT),
                                          ylabel='precision',
                                          title='Binary PR curve on {:s}'.format(
                                              INSECURE_TYPES[i]),
                                          legend='auc: {:.4f}'.format(curves[2]))
                    per_class_prcurves_plots.append(p)
                    with open(SAVE_NPZ + 'test_binary_PRCurve_{:s}.npz'.format(INSECURE_TYPES[i]), 'w') as fd:
                        np.savez(fd,
                                 recall=curves[1],
                                 precision=curves[0],
                                 dropout=DROPOUT)

            # plot average auc on test set
            p6 = vis.simple_curves(range(EPOCHS),
                                   yvalues=res['test_auc'],
                                   legend='AUC of ROC curve',
                                   legend_loc='bottom_right',
                                   title='Avg. Test AUC (averaging over all classes)',
                                   xlabel='epochs')
            best_auc_epoch_avg = np.argmax(res['test_auc'])
            print '>>>> Best AUC epoch == {:d} with avg. score: {:.4f}'.format(best_auc_epoch_avg, res['test_auc'][best_auc_epoch_avg])
            with open(SAVE_NPZ + 'test_auc_roc_avg.npz', 'w') as fd:
                np.savez(fd,
                         epochs=range(EPOCHS),
                         yvalues=res['test_auc'],
                         best_auc=res['test_auc'][best_auc_epoch_avg],
                         best_epoch=best_auc_epoch_avg,
                         legend=['AUC of ROC curve'])

            # plot roc curves per class
            per_class_roc_plots = []
            test_roc_curves = res['test_roc_curves']
            for c in range(num_classes):
                auc_scores = []
                for e in range(EPOCHS):
                    fpr, tpr, thres = test_roc_curves[e][c]
                    auc_scores.append(auc(fpr, tpr))
                best_epoch = np.argmax(auc_scores)
                best_auc = auc_scores[best_epoch]
                fpr, tpr, thres = test_roc_curves[best_epoch][c]
                p_roc = vis.simple_curves(fpr,
                                          yvalues=tpr,
                                          legend=label_names[c],
                                          legend_loc='bottom_right',
                                          title='ROC curve for {:s} type | AUC={:.4f} | Epoch={:d}'.format(
                                                label_names[c], best_auc, best_epoch),
                                          xlabel='FPR',
                                          ylabel='TRP')
                per_class_roc_plots.append(p_roc)
                with open(SAVE_NPZ + 'test_roc_{:s}.npz'.format(label_names[c]), 'w') as fd:
                    np.savez(fd,
                             fpr=fpr,
                             tpr=tpr,
                             thres=thres,
                             auc=best_auc,
                             epoch=best_epoch,
                             legend=[label_names[c]])

            bokeh_grids = gridplot(
                [[p1, p2, p3], [p4, p5], per_class_prcurves_plots, [p6], per_class_roc_plots])
            show(bokeh_grids)
