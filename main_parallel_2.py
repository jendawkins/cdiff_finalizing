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
    parser.add_argument("-model", "--model", help = "model (LR, RF)", type = str)
    parser.add_argument("-test_feat", "--test_feat", help = "feature to test", type = float, nargs = '+')
    parser.add_argument("-final", "--final", help = "final testing based on saved parameters or not", type = bool)

    args = parser.parse_args()
    mb = basic_ml()
    if not args.i:
        path = 'inputs/week_one_metabs/'
    else:
        path = 'inputs/' + args.i + '/'
    with open(path + 'x.pkl','rb') as f:
        x = pkl.load(f)
    with open(path + 'y.pkl','rb') as f:
        y = pkl.load(f)

    coef_names = x.columns.values

    if args.o:
        path_out = args.o + '/'
    else:
        path_out = 'outputs/'

    if not os.path.isdir(path_out):
        os.mkdir(path_out)
    
    if args.i:
        path_out = path_out + args.i + '/'
    else:
        path_out = path_out + 'week_one_metabs/'

    if args.final == True:
        path_out = path_out + 'final/'

    if not os.path.isdir(path_out):
        os.mkdir(path_out)

    seed = args.seed

    try:
        with open(path_out + args.param+ "_" + str(args.seed) + "_" + str(args.ix) + '.pkl','rb') as f:
            final_res_dict = pkl.load(f)
    except:
        final_res_dict = {}
        f1 = open(path_out + "InitializationLogger.txt","a")
        f1.write(args.param + ' ' + str(args.ix) + ' initialized at seed ' + str(args.seed) + '\n')
        f1.close()
    
    if args.ix not in final_res_dict.keys():
        final_res_dict[args.ix] = {}

    if args.model == 'LR':
        model = LogisticRegression(class_weight = 'balanced', penalty = 'l1', random_state = seed, solver = 'liblinear')
        if 'all_data' in args.i:
            model = LogisticRegression(class_weight = None, penalty = 'l1', random_state = seed, solver = 'liblinear')
        lv = 'C'
        feature_grid = np.logspace(-3,3,100)
        ft = args.test_feat[0]

    elif args.model == 'RF':
        model = RandomForestClassifier(class_weight = 'balanced', random_state = seed)
        if 'all_data' in args.i:
            model = RandomForestClassifier(class_weight = None, random_state = seed)
        lv = ['n_estimators','max_depth']
        estimators_grid = np.arange(2,51,2)
        depth_grid = np.arange(2,20,1)
        feature_grid = list(itertools.product(estimators_grid, depth_grid))
        ft = tuple([int(tf) for tf in args.test_feat])

    if args.param == 'coef_bootstrap' or args.param == 'auc':
        seed, X, y = mb.starter(model, x, y, 'metabolites', 'week_one')
        ixs = leave_one_out_cv(x,y)
        train_index, test_index = ixs[args.ix]
        X_train, X_test = x.iloc[train_index, :], x.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        final_res_dict[ft] = mb.one_cv_func(model, X_train, y_train,dtype =None, \
            var_to_learn= lv, test_param = ft)

        if args.final == True:
            res_dict = get_resdict_from_file(path_out)
            best_param_dict = get_best_param(res_dict, args.param)

            train_inde, test_index = ixs[args.ix]
            best_param = best_param_dict[args.ix]
            final_res_dict = mb.fit_all(model, x, y, var_to_learn = lv, test_param = best_param)
            
    if args.param == 'auc_bootstrap':
        seed, X, y = mb.starter(model, x, y, 'metabolites', 'week_one')
        ixs = leave_one_out_cv(x,y)
        train_index, test_index = ixs[args.ix]
        X_train, X_test = x.iloc[train_index, :], x.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        res_dict = mb.nested_cv_func(model, X_train, y_train, dtype = 'metabolites', optim_param = 'auc', plot_lambdas=False, \
            learn_var = lv,feature_grid = feature_grid)
        if args.ix not in final_res_dict[seed].keys():
            final_res_dict[seed][args.ix] = res_dict

    if args.param == 'best_lambda':
        auc = {}
        for l in feature_grid:
            final_res_dict_c2 = mb.one_cv_func(model, x, y,dtype =None, var_to_learn= lv, test_param = l)
            auc[l] = final_res_dict_c2['metrics']['auc']
        m_auc = np.max(list(auc.values()))
        best_l = [l for l,a in auc.items() if a==m_auc]
        final_res_dict = best_l

    if args.param == 'coef':
        param_vec = []
        for file in os.listdir(path_out):
            if 'best_lambda' in file:
                with open(path_out + file,'rb') as f:
                    frd = pickle.load(f)
            param_vec.append(list(frd.values())[0][0])
        
        if args.model == 'RF':
            out = [np.median(list(x)) for x in list(zip(*param_vec))]
            best_param = tuple(out)
        else:
            best_param = np.median(param_vec)
        final_res_dict = mb.fit_all(model, x, y, dtype = 'metabolites', optim_param = 'auc', var_to_learn = lv, optimal_param = best_param)
    
    with open(path_out + args.param+ "_" + str(args.seed) + "_" + str(args.ix) + '.pkl','wb') as f:
        pickle.dump(final_res_dict, f)

    # end = time.time()
    # passed = np.round((end - start)/60, 3)
    # f2 = open(path_out + args.param + ".txt","a")
    # f2.write(str(args.seed) + ' complete at ' + str(args.ix) +  ' in ' + str(passed) + ' minutes' + '\n')
    # f2.close()
        
    


# Get coefficients from nested CV 
