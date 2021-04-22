from dataLoaderCdiff import *
import scipy.stats as st
from collections import Counter
from ml_methods import *
import sklearn
import torch
from matplotlib import cm
import scipy
import  itertools
from datetime import datetime

from seaborn import clustermap
from scipy.cluster.hierarchy import linkage
from sklearn.linear_model import LogisticRegression
import os
import time
import pickle as pkl
from sklearn.ensemble import RandomForestClassifier
import time
from helper import *
from sklearn.model_selection import GridSearchCV
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", "--input", help='title', type=str)
    parser.add_argument("-output", "--output", help='title', type=str)
    parser.add_argument("-seed", "--seed", help='title', type=int)
    parser.add_argument("-ix", "--ix", help='title', type=int)
    args = parser.parse_args()

    path_out = args.output
    if not os.path.isdir(path_out):
        os.mkdir(path_out)
    path_out = path_out + '/' + args.input
    if not os.path.isdir(path_out):
        os.mkdir(path_out)

    paths = os.listdir('inputs/in/')
    dat_dict = {}
    for path in paths:
        if 'DS' in path:
            continue
        with open('inputs/in/' + path + '/x.pkl','rb') as f:
            x = pkl.load(f)
        with open('inputs/in/' + path + '/y.pkl','rb') as f:
            y = pkl.load(f)
        dat_dict[path] = (x,y)

    x, y = dat_dict[args.input]
    y = (np.array(y)=='Recur').astype('float')
    ixs = leave_one_out_cv(x,y)

    n_estimators = [int(x) for x in np.arange(5,50,5)]
    max_features = ['sqrt']
    max_depth = [int(x) for x in np.arange(2,20,1)]
    min_samples_split = [2,4,5]
    min_samples_leaf = [1,3,4]
    bootstrap = [False]

    search_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

    rf = RandomForestClassifier(random_state = args.seed, class_weight = 'balanced')

    results= {}
    # for i,ix in enumerate(ixs):
    ix = ixs[args.ix]
    tr, ts = ix
    x_tr, x_ts = x.iloc[tr,:], x.iloc[ts,:]
    y_tr, y_ts = y[tr], y[ts]
    ixs_tr = leave_one_out_cv(x_tr, y_tr)
    probs_vec = []
    true_vec = []
    best_params = []
    pred_vec = []
    for ix_in in ixs_tr:
        tr_in, ts_in = ix_in
        x_tr2, x_ts2 = x.iloc[tr_in, :], x.iloc[ts_in,:]
        y_tr2, y_ts2 = y[tr_in], y[ts_in]
        
        ixs_cv = leave_one_out_cv(x_tr2, y_tr2)
        rf_grid = GridSearchCV(estimator = rf, param_grid = search_grid, \
                                    cv = ixs_cv, verbose=2, n_jobs = -1, \
                            scoring ='balanced_accuracy')
        rf_grid.fit(x_tr2, y_tr2)
        
        pred_probs = rf_grid.predict_proba(x_ts2)
        pred = rf_grid.predict(x_ts2)
        probs = pred_probs.squeeze()[1]
        probs_vec.append(float(probs))
        true_vec.append(int(y_tr2.squeeze()))
        best_params.append(rf_grid.best_params_)
        pred_vec.append(int(pred.squeeze()))
    results['probs'] = probs_vec
    results['pred'] = pred_vec
    results['true'] = true_vec
    results['best_params'] = best_params
    results['metrics'] = get_metrics(pred_vec, true_vec, probs)

    with open(path_out + '/seed' + str(args.seed) + 'ix' + str(args.ix) + '.pkl','wb') as f:
        pickle.dump(results, f)
    
    f2 = open(path_out + "/res_file" +str(args.seed) + ".txt","a")
    f2.write(str(args.ix) + ' AUC: ' + str(results['metrics']['auc'])+ '\n')
    f2.close()

