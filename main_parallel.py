from basic_ml import *
from helper import *
from datetime import datetime
import pickle as pkl
import pandas as pd
import numpy as np
import argparse
import os
import time

if __name__ == "__main__":

    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("-seed", "--seed", help="random_seed", type=int)
    parser.add_argument("-param", "--param", help = "coef, coef_bootstrap, auc, auc_bootstrap, or best_lambda", type = str)
    parser.add_argument("-ix", "--ix", help = "index for splits", type = int)
    parser.add_argument("-sa", "--sa", help = "smooth auc or not", type = bool)
    parser.add_argument("-o", "--o", help = "outpath", type = str)

    args = parser.parse_args()
    mb = basic_ml()
    path = 'inputs/week_one_metabs/'
    with open(path + 'w1_x.pkl','rb') as f:
        x = pkl.load(f)
    with open(path + 'w1_y.pkl','rb') as f:
        y = pkl.load(f)

    coef_names = x.columns.values

    if args.o:
        path_out = args.o
    else:
        path_out = 'outputs'
    
    if args.sa is True or args.sa is None:
        path_out = path_out + '_smooth/'
    elif args.sa is False:
        path_out = path_out + '_not_smooth/'

    if not os.path.isdir(path_out):
        os.mkdir(path_out)

    try:
        with open(path_out + args.param + '.pkl','rb') as f:
            final_res_dict = pkl.load(f)
    except:
        final_res_dict = {}
        f1 = open("InitializationLogger.txt","a")
        f1.write(args.param + ' ' + str(args.ix) + ' initialized at seed ' + str(args.seed) + '\n')
        f1.close()

    seed = args.seed
    if seed not in final_res_dict.keys():
        final_res_dict[seed] = {}

    model = LogisticRegression(class_weight = 'balanced', penalty = 'l1', random_state = seed, solver = 'liblinear')
    if args.param == 'coef_bootstrap' or args.param == 'auc':
        final_res_dict[seed] = mb.nested_cv_func(model, x, y, dtype = 'metabolites', optim_param = 'auc', plot_lambdas=False, learn_var = 'C')
        
    if args.param == 'coef':
        final_res_dict[seed] = mb.fit_all(model, x, y, dtype = 'metabolites', optim_param = 'auc')

    if args.param == 'auc_bootstrap':
        seed, X, y = mb.starter(model, x, y, 'metabolites', 'week_one')
        ixs = leave_one_out_cv(x,y)
        train_index, test_index = ixs[args.ix]
        X_train, X_test = x.iloc[train_index, :], x.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        res_dict = mb.nested_cv_func(model, X_train, y_train, dtype = 'metabolites', optim_param = 'auc', plot_lambdas=False, learn_var = 'C', smooth_auc=args.sa)
        if args.ix not in final_res_dict[seed].keys():
            final_res_dict[seed][args.ix] = res_dict

    if args.param == 'best_lambda':
        auc = {}
        for l in np.logspace(-3,3,200):
            model = LogisticRegression(C = 1/l, class_weight = 'balanced', penalty = 'l1', \
                                random_state = 0, solver = 'liblinear')
            final_res_dict_c2 = mb.one_cv_func(model, x, y,dtype =None)
            auc[l] = final_res_dict_c2['metrics']['auc']
        m_auc = np.max(list(auc.values()))
        best_l = [l for l,a in auc.items() if a==m_auc]
        final_res_dict[seed] = best_l
    
    with open(path_out + args.param+ '.pkl','wb') as f:
        pickle.dump(final_res_dict, f)
    
    end = time.time()
    passed = np.round((end - start)/60, 3)
    f2 = open(args.param + ".txt","a")
    f2.write(str(args.seed) + ' complete at ' + str(args.ix) +  ' in ' + str(passed) + ' minutes' + '\n')
    f2.close()
        
    


# Get coefficients from nested CV 
