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
import warnings

warnings.filterwarnings("ignore")

def train_lr(x, y, lambda_min_ratio = .0001, path_len = 100, path_out = '', plot_lambdas = False):
    # if feature_grid is None:
    #     feature_grid = np.logspace(7, 20, 14)
    probs = []
    outcomes = []
    model_out_dict = {}
    ix_inner = leave_one_out_cv(x, y)
    lambda_dict = {}
    l_path = np.logspace(0,9,100)
    # m, n = np.shape(x)
    # d_0ii = sigmoid(np.zeros(x.shape[0]))*(1-sigmoid(np.zeros(x.shape[0])))
    # H = (x.T@np.diag(d_0ii))@x
    # w_0 = (x@x.T)@(sigmoid(np.zeros(x.shape[0]))*(1-sigmoid(np.zeros(x.shape[0]))))
    # d_0 = x.T@(sigmoid(np.zeros(x.shape[0])) - y)
    # z_0 = x@np.zeros(x.shape[1]) - (1/w_0)*d_0
    # l_max = np.max([(1/(x.shape[0]))*np.sum(w_0*x.iloc[:,j]*z_0) for j in np.arange(x.shape[1])])
    # l_max = max(list(abs(np.dot(np.transpose(x), y))))/ m
    # l_path = np.linspace(l_max*lambda_min_ratio,l_max, path_len)
    for ic_in, ix_in in enumerate(ix_inner):
        train_index, test_index = ix_in
        x_train, x_test = x.iloc[train_index, :], x.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]

        ix_inner2 = leave_one_out_cv(x_train, y_train)
        lamb_dict = {}
        lamb_dict['auc'] = {}
        lamb_dict['ci'] = {}

        model_dict = {}
        test_probs = {}
        y_true = {}


        if sum(y_train)==1:
            continue
        for ic_in2, ix_in2 in enumerate(ix_inner2):
            start_inner = time.time()

            train_ix, test_ix = ix_in2
            x_tr2, x_ts2 = x_train.iloc[train_ix, :], x_train.iloc[test_ix, :]
            y_tr2, y_ts2 = y_train[train_ix], y_train[test_ix]

            if sum(y_tr2)==0:
                continue

            if ic_in == 0 and ic_in2 == 0:
                for i, lam in enumerate(l_path):
                    model2 = LogisticRegression(penalty='l1', class_weight='balanced', C=1 / lam, solver='liblinear')
                    model2.fit(x_tr2, y_tr2)
                    if (abs(model2.coef_)<=1e-8).all():
                        l_max = lam
                        l_path = np.logspace(np.log10(l_max * lambda_min_ratio), np.log10(l_max), path_len)
                        break

            for i,lam in enumerate(l_path):
                model2 = LogisticRegression(penalty='l1', class_weight='balanced', C = 1/lam, solver = 'liblinear')
                model2.fit(x_tr2, y_tr2)

                if i not in test_probs.keys():
                    test_probs[i]={}
                    model_dict[i]={}
                test_probs[i][ic_in2]= model2.predict_proba(x_ts2)
                model_dict[i][ic_in2] = model2
            y_true[ic_in2] = y_ts2
        scores = {}
        pt_ixs = list(y_true.keys())
        for l_ix in test_probs.keys():
            l_test = l_path[l_ix]
            scores[l_ix] = sklearn.metrics.roc_auc_score([y_true[iix].item() for iix in pt_ixs],
                                                         [test_probs[l_ix][iix][:,1].item() for iix in pt_ixs])

        lambdas, aucs_in = list(zip(*scores.items()))
        if plot_lambdas:
            plt.figure()
            plt.plot([l_path[li] for li in lambdas], aucs_in)
            plt.ylim([0,1])
            plt.xscale('log')
            plt.xlabel('lambdas')
            plt.ylabel('auc')
            plt.savefig(path_out + '/fold' +str(ic_in) + '.pdf')

        ix_max = np.argmax(aucs_in)
        best_lamb = l_path[lambdas[ix_max]]

        lambda_dict[ic_in] = {'best_lambda': best_lamb, 'scores': scores, 'outcomes':y_true,
                       'probs':test_probs, 'lambdas_tested': l_path}
        model_out = LogisticRegression(penalty='l1', class_weight='balanced', C = 1/best_lamb, solver = 'liblinear')
        model_out.fit(x_train, y_train)

        risk_scores = model_out.predict_proba(x_test)

        probs.append(risk_scores[:,1].item())
        outcomes.append(y_test.item())

        model_out_dict[ic_in] = model_out

    score = sklearn.metrics.roc_auc_score(outcomes, probs)

    final_dict = {}
    final_dict['auc'] = score
    final_dict['model'] = model_out_dict
    final_dict['probs'] = probs
    final_dict['outcomes'] = outcomes
    final_dict['lambdas'] = lambda_dict
    return final_dict


if __name__ == "__main__":

    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("-ix", "--ix", help="index for splits", type=int)
    parser.add_argument("-o", "--o", help="outpath", type=str)
    parser.add_argument("-i", "--i", help="inpath", type=str)
    parser.add_argument("-type", "--type", help="inpath", type=str)
    parser.add_argument("-week", "--week", help="week", type=float)
    args = parser.parse_args()
    mb = basic_ml()

    if not args.ix:
        args.ix = 7
        args.o = 'test_lr_fast'
        if not os.path.isdir(args.o):
            os.mkdir(args.o)
        args.i = 'metabs'
        args.type = 'auc'
        args.week = [0,1.5,2]

    if args.i == '16s':
        dl = dataLoader(pt_perc=.05, meas_thresh=10, var_perc=5, pt_tmpts=1)
    else:
        dl = dataLoader(pt_perc=.25, meas_thresh=0, var_perc=15, pt_tmpts=1)

    if isinstance(args.week, list):
        x, y, event_times = get_slope_data(dl.week[args.i], args.week)
    else:
        data = dl.week[args.i][args.week]
        x, y, event_times = data['x'], data['y'], data['event_times']
        x.index = [xind.split('-')[0] for xind in x.index.values]

    y = (np.array(y)=='Recurrer').astype('float')

    path_out = args.o + '/' + args.i + '/'

    if not os.path.isdir(path_out):
        os.mkdir(path_out)

    if args.type == 'auc':
        ixs = leave_one_out_cv(x, y)
        args.ix = np.where(y==1)[0][0]
        train_index, test_index = ixs[args.ix]
        x_train0, x_test0 = x.iloc[train_index, :], x.iloc[test_index, :]
        y_train0, y_test0 = y[train_index], y[test_index]

        final_res_dict = train_lr(x_train0, y_train0, path_out = path_out)
    elif args.type == 'coef':
        final_res_dict = train_lr(x,y, path_out = path_out)

    final_res_dict['data'] = (x, y)

    with open(path_out + args.type + '_ix_' + str(args.ix)+ '.pkl', 'wb') as f:
        pickle.dump(final_res_dict, f)

    end = time.time()
    passed = np.round((end - start) / 60, 3)
    f2 = open(args.o + '/' + args.i + ".txt", "a")
    f2.write('index ' + str(args.ix) + ', CI ' + str(final_res_dict['auc']) +' in ' + str(passed) + ' minutes' + '\n')
    f2.close()
