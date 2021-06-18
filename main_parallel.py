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
from dataLoader import *
from sklearn.linear_model import LogisticRegression


if __name__ == "__main__":

    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("-seed", "--seed", help="random_seed", type=int)
    parser.add_argument("-type", "--type", help = "coef or auc", type = str)
    parser.add_argument("-ix", "--ix", help = "index for splits", type = int)
    parser.add_argument("-o", "--o", help = "outpath", type = str)
    parser.add_argument("-i", "--i", help = "inpath", type = str)
    parser.add_argument("-week", "--week", help="week", type=str)
    args = parser.parse_args()
    mb = basic_ml()

    if not args.i:
        args.type = 'auc'
        args.ix = 0

        args.i = 'metabs'
        args.week = 1.5
        args.o = 'test' + '_'.join(str(args.week).split('.'))
        if not os.path.isdir(args.o):
            os.mkdir(args.o)
    else:
        args.week = [float(w) for w in args.week.split('_')]

    if args.i == '16s':
        dl = dataLoader(pt_perc = .05, meas_thresh = 10, var_perc = 5, pt_tmpts = 1)
    else:
        dl = dataLoader(pt_perc=.25, meas_thresh=0, var_perc=15, pt_tmpts=1)
    if isinstance(args.week, list):
        x, y, event_times = get_slope_data(dl.week[args.i], args.week)
    else:
        data = dl.week[args.i][args.week]
        x, y, event_times = data['x'], data['y'], data['event_times']

    x.index = [xind.split('-')[0] for xind in x.index.values]

    if not args.seed:
        args.seed = 0


    if isinstance(y[0], str):
        y = (np.array(y) == 'Recurrer').astype('float')
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
    feature_grid = np.logspace(-7,2,300)

    if args.type == 'coef':
        final_res_dict = mb.nested_cv_func(model, x, y, optim_param='auc', plot_lambdas=False, learn_var=lv, \
                                           feature_grid=feature_grid, model_2 = None, smooth_auc = False)

    if args.type == 'auc':
        ixs = leave_one_out_cv(x,y)
        train_index, test_index = ixs[args.ix]
        X_train, X_test = x.iloc[train_index, :], x.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        final_res_dict = mb.nested_cv_func(model, X_train, y_train,optim_param = 'auc', plot_lambdas=False, learn_var = lv, \
            feature_grid = feature_grid, model_2 = None)
    
    if 'auc' in args.type:
        with open(path_out + args.type+ "_" + str(args.seed) + "_" + str(args.ix) + '.pkl','wb') as f:
            pkl.dump(final_res_dict, f)
    else:
        with open(path_out + "_" + args.type+ "_" + str(args.seed) + '.pkl','wb') as f:
            pkl.dump(final_res_dict, f)
    
    end = time.time()
    passed = np.round((end - start)/60, 3)
    f2 = open(args.o + '/' + args.i + ".txt","a")
    f2.write('Seed ' + str(args.seed) + ', index ' + str(args.ix) + ', AUC: ' + str(final_res_dict['metrics']['auc']) \
             + ' in ' + str(passed) + ' minutes' + '\n')
    f2.close()
        
    


# Get coefficients from nested CV 
