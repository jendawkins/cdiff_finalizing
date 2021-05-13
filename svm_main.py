import scipy.stats as st
from collections import Counter
import sklearn
import torch
from matplotlib import cm
import scipy
import itertools
from datetime import datetime
import os
import time
import sys
import pickle as pkl
from helper import *
import argparse
from basic_ml import *



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-output", "--output", help='title', type=str)
    parser.add_argument("-input","--input", type = str)
    parser.add_argument("-seed", "--seed", type=int)
    parser.add_argument("-ix", "--ix", type=int)
    args = parser.parse_args()

    if not args.output:
        args.output = 'test'
    if not args.input:
        args.input = 'week_one_metabs'
    if not args.seed:
        args.seed = 0
    if not args.ix:
        args.ix = 0

    model = sklearn.svm.SVC(class_weight='balanced', random_state = args.seed, probability = False)

    mb = basic_ml()
    if not args.output:
        print('Specify out dir')
        sys.exit(1)

    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    path_out = args.output + '/' +  args.input + '/'
    if not os.path.isdir(path_out):
        os.mkdir(path_out)

    lv = 'C'
    feature_grid = np.logspace(-7, 3, 200)

    path = 'inputs/in_15/' + args.input + '/'
    with open(path + 'x.pkl', 'rb') as f:
        x = pkl.load(f)
    with open(path + 'y.pkl', 'rb') as f:
        y = pkl.load(f)

    if isinstance(y[0], str):
        y = (np.array(y) == 'Recur').astype('float')
    else:
        y = np.array(y)

    ixs = leave_one_out_cv(x, y)
    train_index, test_index = ixs[args.ix]
    X_train, X_test = x.iloc[train_index, :], x.iloc[test_index, :]
    y_train, y_test = y[train_index], y[test_index]
    start = time.time()
    final_res_dict = mb.nested_cv_func(model, X_train, y_train, optim_param='f1', plot_lambdas=False, learn_var=lv, \
                                       feature_grid=feature_grid)

    end = time.time()
    print(end - start)
    with open(path_out + "seed" +  str(args.seed) + "_ix" + str(args.ix) + '.pkl', 'wb') as f:
        pickle.dump(final_res_dict, f)

    f2 = open(args.output + "/" + args.input + "_SVM_seed" + str(args.seed) +  ".txt","a")
    f2.write('Index ' + str(args.ix) +  ', AUC= ' + str(final_res_dict['metrics']['auc']) + ', Time = ' + str(end - start) + '\n')
    f2.close()