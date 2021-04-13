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

    args = parser.parse_args()
    mb = basic_ml()
    if not args.i:
        path = 'inputs/in/week_one_metabs/'
    else:
        path = 'inputs/in/' + args.i + '/'
    with open(path + 'x.pkl','rb') as f:
        x = pkl.load(f)
    with open(path + 'y.pkl','rb') as f:
        y = pkl.load(f)

    if isinstance(y[0], str):
        y = (np.array(y) == 'Recur').astype('float')
    else:
        y = np.array(y)

    coef_names = x.columns.values

    if args.o:
        path_out = args.o + '/'
    else:
        path_out = 'outputs/'

    if not os.path.isdir(path_out):
        os.mkdir(path_out)
    
    path_out = path_out + args.model + '/'
    if not os.isdir(path_out):
        os.mkdir(path_out)
        
    if args.i:
        path_out = path_out + args.i + '/'
    else:
        path_out = path_out + 'week_one_metabs/'

    if not os.path.isdir(path_out):
        os.mkdir(path_out)

    final_res_dict = {}

    seed = args.seed
    # if seed not in final_res_dict.keys():
    #     final_res_dict[seed] = {}

    if args.model == 'LR':
        model = LogisticRegression(class_weight = 'balanced', penalty = 'l1', random_state = seed, solver = 'liblinear')
        if 'all_data' in args.i:
            model = LogisticRegression(class_weight = None, penalty = 'l1', random_state = seed, solver = 'liblinear')
        lv = 'C'
        feature_grid = np.logspace(-3,3,100)

    elif args.model == 'RF':
        model = RandomForestClassifier(class_weight = 'balanced', random_state = seed)
        if 'all_data' in args.i:
            model = RandomForestClassifier(class_weight = None, random_state = seed)
        lv = ['n_estimators','max_depth']
        estimators_grid = np.arange(2,51,2)
        estimators_grid = np.append(estimators_grid, np.arange(55,100,5))
        depth_grid = np.arange(2,20,1)
        feature_grid = list(itertools.product(estimators_grid, depth_grid))

    # if args.param == 'coef_bootstrap' or args.param == 'auc':
    #     final_res_dict[seed] = mb.nested_cv_func(model, x, y,optim_param = 'auc', plot_lambdas=False, learn_var = lv, \
    #         feature_grid = feature_grid)
        

    if args.param == 'coef_bootstrap' or args.param == 'auc':
        ixs = leave_one_out_cv(x,y)
        train_index, test_index = ixs[args.ix[0]]
        X_train, X_test = x.iloc[train_index, :], x.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        res_dict = mb.nest_cv_func(model, X_train, y_train, optim_param = 'auc', plot_lambdas=False, \
            learn_var = lv,feature_grid = feature_grid)
        final_res_dict = res_dict

    if args.param == 'auc_bootstrap':
        ixs = leave_one_out_cv(x,y)
        train_index, test_index = ixs[args.ix]
        X_train, X_test = x.iloc[train_index, :], x.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        final_res_dict = mb.nested_cv_func(model, X_train, y_train,optim_param = 'auc', plot_lambdas=False, learn_var = lv, \
            feature_grid = feature_grid)

    # if args.param == 'auc_bootstrap':
    #     ixs = leave_one_out_cv(x,y)
    #     train_index, test_index = ixs[args.ix[0]]
    #     X_train, X_test = x.iloc[train_index, :], x.iloc[test_index, :]
    #     y_train, y_test = y[train_index], y[test_index]

    #     ixs = leave_one_out_cv(X_train,y_train)
    #     train_index_in, test_index_in = ixs[args.ix[1]]
    #     X_train_in, X_test_in = X_train.iloc[train_index_in,:], X_train.iloc[test_index_in]
    #     y_train_in, y_test_in = y_train[train_index_in], y_train[test_index_in]
    #     res_dict_inner = mb.nest_cv_func(model, X_train_in, y_train_in, optim_param = 'auc', plot_lambdas=False, \
    #         learn_var = lv,feature_grid = feature_grid)

    #     best_param = res_dict_inner['best_lambda']
    #     res_dict_outer = mb.train_test(model, X_train, X_test, y_train, y_test, optimal_param = best_param, learn_var = lv)
    #     # if args.ix[0] not in final_res_dict[seed].keys():
    #     #     final_res_dict[seed][args.ix[0]] = {}
    #     final_res_dict['outer_metrics'] = res_dict_outer
    #         # if args.ix[1] not in final_res_dict[seed][args.ix[0]].keys():
    #     final_res_dict['inner_metrics'] = res_dict_inner

    if args.param == 'best_lambda':
        auc = {}
        for l in feature_grid:
            final_res_dict_c2 = mb.one_cv_func(model, x, y,dtype =None, var_to_learn= lv, test_param = l)
            auc[l] = final_res_dict_c2['metrics']['auc']
        m_auc = np.max(list(auc.values()))
        best_l = [l for l,a in auc.items() if a==m_auc]
        final_res_dict[seed] = best_l

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
        final_res_dict[seed] = mb.fit_all(model, x, y, optim_param = 'auc', var_to_learn = lv, optimal_param = best_param)
    
    if 'auc_bootstrap' in args.param:
        with open(path_out + "_" + args.param+ "_" + str(args.seed) + "_" + str(args.ix) + '.pkl','wb') as f:
            pickle.dump(final_res_dict, f)
    else:
        with open(path_out + "_" + args.param+ "_" + str(args.seed) + '.pkl','wb') as f:
            pickle.dump(final_res_dict, f)
    
    end = time.time()
    passed = np.round((end - start)/60, 3)
    f2 = open(args.param + args.model + ".txt","a")
    f2.write(str(args.seed) + ' complete at ' + str(args.ix) +  ' in ' + str(passed) + ' minutes' + '\n')
    f2.close()
        
    


# Get coefficients from nested CV 
