from basic_ml import *
from helper import *
from datetime import datetime
import pickle as pkl
import pandas as pd
import numpy as np
import argparse
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-seed", "--seed", help="random_seed", type=int)
    parser.add_argument("-param", "--param", help = "coef, coef_bootstrap, auc, auc_bootstrap, or best_lambda", type = str)

    args = parser.parse_args()
    mb = basic_ml()
    path = 'inputs/week_one_metabs/'
    with open(path + 'w1_x.pkl','rb') as f:
        x = pkl.load(f)
    with open(path + 'w1_y.pkl','rb') as f:
        y = pkl.load(f)

    coef_names = x.columns.values

    path_out = 'outputs/'+ str(datetime.now()).split(' ')[0] + '/'
    if not os.path.isdir(path_out):
        os.mkdir(path_out)

    try:
        with open(path_out + args.param + '.pkl','rb') as f:
            final_res_dict = pkl.load(f)
    except:
        final_res_dict = {}
        print('initialized at seed ' + str(args.seed))

    seed = args.seed

    model = LogisticRegression(class_weight = 'balanced', penalty = 'l1', random_state = seed, solver = 'liblinear')
    if args.param == 'coef_bootstrap' or args.param == 'auc':
        final_res_dict[seed] = mb.nested_cv_func(model, x, y, dtype = 'metabolites', optim_param = 'auc', plot_lambdas=False, learn_var = 'C')
        
    if args.param == 'coef':
        final_res_dict[seed] = mb.fit_all(model, x, y, dtype = 'metabolites', optim_param = 'auc')

    if args.param == 'auc_bootstrap':
        final_res_dict[seed] = mb.double_nest(model, x, y, dtype = 'metabolites', optim_param = 'auc', plot_lambdas=False, learn_var = 'C')

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
        
    


# Get coefficients from nested CV 
