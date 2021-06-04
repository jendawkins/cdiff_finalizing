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
from lifelines import CoxPHFitter
import warnings
from lifelines.utils import concordance_index
warnings.filterwarnings("ignore")

def train_cox(x_train0, ix_in, y_per_pt, y_int, metric = 'auc', feature_grid = None):
    if feature_grid is None:
        feature_grid = np.logspace(7, 20, 14)
    survival = {}
    # for ic_in, ix_in in enumerate(ix_inner):
    train_index, test_index = ix_in
    x_train, x_test = x_train0.iloc[train_index, :], x_train0.iloc[test_index, :]

    lamb_dict = {}
    for lamb in feature_grid:
        ix_inner2 = leave_one_out_cv(x_train, x_train['outcome'], ddtype='all_data')
        # ix_inner2_rand_samp = np.random.choice(ix_inner2, 10, replace = False)
        counter = 0
        start = time.time()

        hazards = []
        event_times = []
        event_outcomes = []
        probs_in = []
        true = []

        model = CoxPHFitter(penalizer=lamb, l1_ratio=1.)
        for ic_in2, ix_in2 in enumerate(ix_inner2):
            start_inner = time.time()

            train_ix, test_ix = ix_in2
            x_tr2, x_ts2 = x_train.iloc[train_ix, :], x_train.iloc[test_ix, :]
            tmpts_in = [xx.split('-')[1] for xx in x_tr2.index.values]
            samp_weights = get_class_weights(np.array(y_int[x_tr2.index.values]), tmpts_in)
            samp_weights[samp_weights <= 0] = 1
            x_tr2.insert(x_tr2.shape[1], 'weights', samp_weights)
            try:
                model.fit(x_tr2, duration_col='week', event_col='outcome',
                          weights_col='weights', robust=False, show_progress = False)
            except:
                counter += 1
                continue
            pred_f = model.predict_survival_function(x_ts2.iloc[0, :])
            probs_in.append(1 - pred_f.loc[4.0].item())
            true.append(x_ts2['outcome'].iloc[-1])
            hazard = model.predict_partial_hazard(x_ts2)
            hazards.append(hazard)
            event_times.append(x_ts2['week'])
            event_outcomes.append(x_ts2['outcome'])
            end_inner = time.time()
            print('Inner ix ' + str(ic_in2) + ' complete in ' + str(end_inner - start_inner))

        if metric == 'CI':
            try:
                score = concordance_index(pd.concat(event_times), pd.concat(hazards), pd.concat(event_outcomes))
                lamb_dict[lamb] = score
                end_t = time.time()
                print(str(lamb) + ' complete')
                print(start - end_t)
            except:
                print('No score available')
                continue
        elif metric == 'auc':
            try:
                score = sklearn.metrics.roc_auc_score(true, probs_in)
                lamb_dict[lamb] = score
                end_t = time.time()
                print(str(lamb) + ' complete')
                print(start - end_t)
            except:
                continue

    lambdas, aucs_in = list(zip(*lamb_dict.items()))
    ix_max = np.argmax(aucs_in)
    best_lamb = lambdas[ix_max]

    model_out = CoxPHFitter(penalizer=best_lamb, l1_ratio=1.)
    tmpts_in = [xx.split('-')[1] for xx in x_train.index.values]
    samp_weights = get_class_weights(np.array(y_int[x_train.index.values]), tmpts_in)
    samp_weights[samp_weights<=0] = 1
    x_train.insert(x_train.shape[1], 'weights', samp_weights)
    x_train['weights'] = samp_weights
    try:
        model_out.fit(x_train, duration_col='week', event_col='outcome', weights_col='weights', robust=False)
    except:
        return {}
    pred_f = model_out.predict_survival_function(x_test.iloc[0, :])
    pt = x_test.index.values[0].split('-')[0]

    hazard_out = model_out.predict_partial_hazard(x_test)


    pts = [ii.split('-')[0] for ii in x.index.values]
    tmpts = [ii.split('-')[1] for ii in x.index.values]
    # if pt not in survival.keys():
        # survival[pt] = {}
    ixs = np.where(np.array(pts) == pt)[0]
    survival['actual'] = str(np.max([float(tmpt) for tmpt in np.array(tmpts)[ixs]]))
    if y_per_pt[pt] == 'Cleared':
        survival['actual'] = survival['actual'] + '+'

    probs_sm = 1 - pred_f.loc[4.0].item()

    y_pred_exp = model_out.predict_expectation(x_test.iloc[[0], :])
    survival['predicted'] = str(np.round(y_pred_exp.item(), 3))
    surv_func = pred_f

    # probs_df = pd.Series(probs_sm)
    # y_pp = y_per_pt.replace('Cleared', 0).replace('Recur', 1)
    # final_df = pd.concat([y_pp, probs_df], axis=1).dropna()

    final_dict = {}
    # final_dict['probability_df'] = final_df
    final_dict['model'] = model_out
    final_dict['survival'] = survival
    final_dict['survival_function'] = surv_func
    final_dict['prob_true'] = (probs_sm, y_per_pt[pt])
    final_dict['times_hazards_outcomes'] = (x_test['week'], hazard_out, x_test['outcome'])
    # final_dict['auc'] = sklearn.metrics.roc_auc_score(final_df[0], final_df[1])
    return final_dict


if __name__ == "__main__":

    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("-ix", "--ix", help="index for splits", type=int, nargs='+')
    parser.add_argument("-o", "--o", help="outpath", type=str)
    parser.add_argument("-i", "--i", help="inpath", type=str)
    parser.add_argument("-metric", "--metric", help="metric", type=str)
    args = parser.parse_args()
    mb = basic_ml()

    if not args.ix:
        args.ix = [0, 0]
        args.o = 'test_cox_preds'
        args.i = 'metabs'
        args.metric = 'auc'
        feature_grid = np.logspace(7, 20, 14)

    # if args.i == '16s':
    #     dl = dataLoader(pt_perc=.05, meas_thresh=10, var_perc=5, pt_tmpts=1)
    # else:
    dl = dataLoader(pt_perc=.25, meas_thresh=0, var_perc=15, pt_tmpts=1)
    y_per_pt = dl.cdiff_data_dict['targets_by_pt']
    x_o = dl.cdiff_data_dict['filtered_data']
    ix_keep = [ix for ix in x_o.index.values if ix.split('-')[1].split('.')[0].isnumeric()]
    x = x_o.loc[ix_keep, :]
    y = dl.cdiff_data_dict['targets'].loc[ix_keep]

    y_val = y.copy()
    pts = [yy.split('-')[0] for yy in y.index.values]
    tmpts = [yy.split('-')[1] for yy in y.index.values]
    for pt in np.unique(pts):
        ix = np.where(np.array(pts) == pt)[0]
        pt_tmpts = np.array(tmpts)[ix]
        p_ix = y.index.values[ix]
        if y_per_pt[pt] == 'Non-recurrer':
            continue
        else:
            y_val[p_ix[-1]] = 'Recurrer'

    x['week'] = tmpts
    x['outcome'] = (np.array(y_val) == 'Recurrer').astype(float)
    y_int = y.replace('Recurrer',1).replace('Non-recurrer',0)

    path_out = args.o + '/' + args.i + '/'

    if not os.path.isdir(path_out):
        os.mkdir(path_out)

    ixs = leave_one_out_cv(x, x['outcome'], ddtype = 'all_data')
    train_index, test_index = ixs[args.ix[0]]
    x_train0, x_test0 = x.iloc[train_index, :], x.iloc[test_index, :]
    ix_inner = leave_one_out_cv(x_train0, x_train0['outcome'], ddtype='all_data')
    final_res_dict = train_cox(x_train0, ix_inner[args.ix[1]], y_per_pt, y_int, metric = args.metric, feature_grid = feature_grid)
    final_res_dict['data'] = x

    with open(path_out + 'ix_' + str(args.ix[0]) +'ix_' + str(args.ix[1])+ '.pkl', 'wb') as f:
        pickle.dump(final_res_dict, f)

    end = time.time()
    passed = np.round((end - start) / 60, 3)
    f2 = open(args.o + '/' + args.i + ".txt", "a")
    f2.write('index ' + str(args.ix) + ' in ' + str(passed) + ' minutes' + '\n')
    f2.close()

# Get coefficients from nested CV
