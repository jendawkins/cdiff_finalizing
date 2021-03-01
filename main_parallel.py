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
    parser.add_argument("-o", "--o", help = "outpath", type = str)
    parser.add_argument("-i", "--i", help = "inpath", type = str)

    args = parser.parse_args()
    mb = basic_ml()
    if not args.i:
        path = 'inputs/week_one_metabs/'
    else:
        path = 'inputs/' + args.i + '/'
    with open(path + 'x2.pkl','rb') as f:
        x = pkl.load(f)
    with open(path + 'y.pkl','rb') as f:
        y = pkl.load(f)

    coef_names = x.columns.values

    if args.o:
        path_out = args.o + '/'
    else:
        path_out = 'outputs/'
    
    if args.i:
        path_out = path_out + args.i + '/'
    else:
        path_out = path_out + 'week_one_metabs/'
    

    if not os.path.isdir(path_out):
        os.mkdir(path_out)

    # try:
    #     with open(path_out + args.param + '.pkl','rb') as f:
    #         final_res_dict = pkl.load(f)
    # except:
    #     final_res_dict = {}
    #     f1 = open("InitializationLogger.txt","a")
    #     f1.write(args.param + ' ' + str(args.ix) + ' initialized at seed ' + str(args.seed) + '\n')
    #     f1.close()

    final_res_dict = {}

    seed = args.seed
    if seed not in final_res_dict.keys():
        final_res_dict[seed] = {}

    model = LogisticRegression(class_weight = 'balanced', penalty = 'l1', random_state = seed, solver = 'liblinear')
    if 'all_data' in args.i:
        model = LogisticRegression(class_weight = None, penalty = 'l1', random_state = seed, solver = 'liblinear')
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
        res_dict = mb.nested_cv_func(model, X_train, y_train, dtype = 'metabolites', optim_param = 'auc', plot_lambdas=False, learn_var = 'C')
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
    
    if args.param == 'auc_bootstrap':
        with open(path_out + args.param+ "_" + str(args.seed) + "_" + str(args.ix) + '.pkl','wb') as f:
            pickle.dump(final_res_dict, f)
    else:
        with open(path_out + args.param+ "_" + str(args.seed) + '.pkl','wb') as f:
            pickle.dump(final_res_dict, f)
    
    end = time.time()
    passed = np.round((end - start)/60, 3)
    f2 = open(args.param + ".txt","a")
    f2.write(str(args.seed) + ' complete at ' + str(args.ix) +  ' in ' + str(passed) + ' minutes' + '\n')
    f2.close()
        
    


# Get coefficients from nested CV 
