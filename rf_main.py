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
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", "--input", help='title', type=str)
    parser.add_argument("-output", "--output", help='title', type=str)
    parser.add_argument("-seed", "--seed", help='title', type=int)
    parser.add_argument("-ix", "--ix", help='title', type=int, nargs = '+')
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

    n_estimators = [int(x) for x in np.arange(5,100,5)]
    min_samples_split = [int(x) for x in np.arange(2,20,1)]

    search_grid = {'n_estimators': n_estimators,
                'min_samples_split': min_samples_split}

    rf = RandomForestClassifier(random_state = args.seed, class_weight = 'balanced', bootstrap=True, max_features = None)

    results= {}
    # for i,ix in enumerate(ixs):
    ix = ixs[args.ix[0]]
    tr, ts = ix
    x_tr, x_ts = x.iloc[tr,:], x.iloc[ts,:]
    y_tr, y_ts = y[tr], y[ts]
    ixs_tr = leave_one_out_cv(x_tr, y_tr)

    ix_in = ixs_tr[args.ix[1]]

    tr_in, ts_in = ix_in
    x_tr2, x_ts2 = x.iloc[tr_in, :], x.iloc[ts_in,:]
    y_tr2, y_ts2 = y[tr_in], y[ts_in]

    ixs_cv = leave_one_out_cv(x_tr2, y_tr2)
    rf_grid = GridSearchCV(estimator = rf, param_grid = search_grid, \
                                cv = ixs_cv, n_jobs = -1, \
                        scoring ='balanced_accuracy')
    rf_grid.fit(x_tr2, y_tr2)

    pred_probs = rf_grid.predict_proba(x_ts2)
    pred = rf_grid.predict(x_ts2)
    probs = pred_probs.squeeze()[1]

    results['probs'] = pred_probs
    results['pred'] = pred
    results['true'] = y_ts2
    results['best_params'] = rf_grid.best_params_
    with open(path_out + '/seed' + str(args.seed) + 'ix_out' + str(args.ix[0]) + 'ix_in' + str(args.ix[1]) + '.pkl','wb') as f:
        pickle.dump(results, f)

    time_out = time.time() - start
    f2 = open(path_out + "/res_file" +str(args.seed) + ".txt","a")
    f2.write(str(args.ix) + ', pred: ' + str(pred_probs[:,1].squeeze()) + ', true: ' + str(y_ts2) + ', time: '+ str(time_out) + '\n')
    f2.close()

