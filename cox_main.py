from basic_ml import *
from helper import *
from datetime import datetime
import pickle as pkl
import pandas as pd
import numpy as np
import argparse
import os
import time
import itertools

if __name__ == "__main__":

    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("-seed", "--seed", help="random_seed", type=int)
    parser.add_argument("-param", "--param", help = "coef, coef_bootstrap, auc, auc_bootstrap, or best_lambda", type = str)
    parser.add_argument("-ix", "--ix", help = "index for splits", type = int)
    parser.add_argument("-o", "--o", help = "outpath", type = str)
    parser.add_argument("-i", "--i", help = "inpath", type = str)
    args = parser.parse_args()
    mb = basic_ml()
    path = 'inputs/in_15/' + args.i + '/'
    with open(path + 'x.pkl','rb') as f:
        x = pkl.load(f)
    with open(path + 'y.pkl','rb') as f:
        y = pkl.load(f)

    if isinstance(y[0], str):
        y = (np.array(y) == 'Recur').astype('float')
    else:
        y = np.array(y)

    coef_names = x.columns.values

    path_out = args.o + '/' + args.i + '/'

    if not os.path.isdir(path_out):
        os.mkdir(path_out)

    final_res_dict = {}

    seed = args.seed
    # if seed not in final_res_dict.keys():
    #     final_res_dict[seed] = {}

    model = LogisticRegression(class_weight = 'balanced', penalty = 'l1', random_state = seed, solver = 'liblinear')
    lv = 'C'
    feature_grid = np.logspace(-3,3,100)
        

    if args.param == 'coef_bootstrap' or args.param == 'auc':
        final_res_dict = mb.nested_cv_func(model, x, y, optim_param='auc', plot_lambdas=False, learn_var=lv, \
                                           feature_grid=feature_grid)

        # ixs = leave_one_out_cv(x,y)
        # train_index, test_index = ixs[args.ix[0]]
        # X_train, X_test = x.iloc[train_index, :], x.iloc[test_index, :]
        # y_train, y_test = y[train_index], y[test_index]
        # res_dict = mb.nest_cv_func(model, X_train, y_train, optim_param = 'auc', plot_lambdas=False, \
        #     learn_var = lv,feature_grid = feature_grid)
        # final_res_dict = res_dict

    if args.param == 'auc_bootstrap':
        ixs = leave_one_out_cv(x,y)
        train_index, test_index = ixs[args.ix]
        X_train, X_test = x.iloc[train_index, :], x.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        final_res_dict = mb.nested_cv_func(model, X_train, y_train,optim_param = 'auc', plot_lambdas=False, learn_var = lv, \
            feature_grid = feature_grid)
    
    if 'auc_bootstrap' in args.param:
        with open(path_out + "_" + args.param+ "_" + str(args.seed) + "_" + str(args.ix) + '.pkl','wb') as f:
            pickle.dump(final_res_dict, f)
    else:
        with open(path_out + "_" + args.param+ "_" + str(args.seed) + '.pkl','wb') as f:
            pickle.dump(final_res_dict, f)
    
    end = time.time()
    passed = np.round((end - start)/60, 3)
    f2 = open(args.o + '/' + args.i + ".txt","a")
    f2.write('Seed ' + str(args.seed) + ', index ' + str(args.ix) + ', AUC: ' + str(final_res_dict['metrics']['auc']) \
             + ' in ' + str(passed) + ' minutes' + '\n')
    f2.close()
        
    


# Get coefficients from nested CV 