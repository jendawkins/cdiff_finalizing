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
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV

warnings.filterwarnings("ignore")

def train_lr_folds(x, y, lambda_min_ratio = .001, path_len = 200, num_folds = 5):

    num_rec = np.sum(y)
    if num_folds > num_rec:
        num_folds = num_rec

    skf = StratifiedKFold(n_splits = num_folds)
    splits = skf.split(x, y)
    score_vec = []
    final_res_dict = {}
    fold = 0


    for train_index, test_index in splits:
        #     probs[ic] = []
        #     train_index, test_index = ix
        x_train, x_test = x.iloc[train_index, :], x.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]

        num_rec = np.sum(y_train)
        if num_folds > num_rec:
            nf_inner = num_rec
        else:
            nf_inner = num_folds

        bval = True
        lam = 0
        while(bval):
            lam += 0.1
            model2 = LogisticRegression(penalty='l1', class_weight='balanced', C=1 / lam, solver='liblinear')
            model2.fit(x_train, y_train)
            if np.sum(np.abs(model2.coef_))<1e-8:
                l_max = lam + 1
                bval = False
        lambda_grid = np.logspace(np.log10(l_max * lambda_min_ratio), np.log10(l_max), path_len)
        lr = LogisticRegression(penalty='l1', class_weight='balanced', solver='liblinear')
        cv = StratifiedKFold(n_splits=nf_inner, shuffle=True, random_state=0)
        clf = GridSearchCV(lr, {'C': lambda_grid}, cv = cv).fit(x_train, y_train)

        cv_results = pd.DataFrame(clf.cv_results_)
        best_model = clf.best_estimator_
        best_coefs = pd.DataFrame(
            best_model.coef_.T,
            index=x_train.columns.values,
            columns=["coefficient"]
        )

        pred_prob = best_model.predict_proba(x_test)
        auc = sklearn.metrics.roc_auc_score(y_test, pred_prob[:,1])
        score_vec.append(auc)
        final_res_dict[fold] = {}
        final_res_dict[fold]['score'] = auc
        final_res_dict[fold]['best_model'] = best_model
        final_res_dict[fold]['best_coefs'] = best_coefs
        final_res_dict[fold]['train_test'] = (x_train, x_test)
        fold += 1
    return final_res_dict


def train_lr(x, y, path_len = 300, path_out = '', plot_lambdas = False):
    if x.shape[0] < x.shape[1]:
        lambda_min_ratio = 0.0001
    else:
        lambda_min_ratio = 0.01
    # if feature_grid is None:
    #     feature_grid = np.logspace(7, 20, 14)
    probs = []
    outcomes = []
    model_out_dict = {}
    ix_inner = leave_one_out_cv(x, y)
    lambda_dict = {}

    for ic_in, ix_in in enumerate(ix_inner):
        train_index, test_index = ix_in
        x_train, x_test = x.iloc[train_index, :], x.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]

        ix_inner2 = leave_one_out_cv(x_train, y_train)
        lamb_dict = {}
        lamb_dict['score'] = {}

        model_dict = {}
        test_probs = {}
        y_true = {}


        if sum(y_train)==1:
            continue

        bval = True
        lam = 0
        while(bval):
            lam += 0.1
            model2 = LogisticRegression(penalty='l1', class_weight='balanced', C=1 / lam, solver='liblinear')
            model2.fit(x_train, y_train)
            if np.sum(np.abs(model2.coef_))<1e-8:
                l_max = lam + 1
                bval = False
        l_path = np.logspace(np.log10(l_max * lambda_min_ratio), np.log10(l_max), path_len)
        for ic_in2, ix_in2 in enumerate(ix_inner2):
            start_inner = time.time()

            train_ix, test_ix = ix_in2
            x_tr2, x_ts2 = x_train.iloc[train_ix, :], x_train.iloc[test_ix, :]
            y_tr2, y_ts2 = y_train[train_ix], y_train[test_ix]

            if sum(y_tr2)==0:
                continue

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

        ma = moving_average(aucs_in, n=5)
        offset = int(np.floor(5. / 2))
        ix_max = np.argmax(ma) + offset
        # ix_max = np.argmax(aucs_in)
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
    final_dict['score'] = score
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
    parser.add_argument("-week", "--week", help="week", type=str)
    parser.add_argument("-folds", "--folds", help="use_folds", type=str)
    parser.add_argument("-num_folds", "--num_folds", help="num_folds", type=int)
    args = parser.parse_args()
    mb = basic_ml()

    if args.ix is None:
        args.ix = 0
    if args.o is None:
        args.o = 'test_lr_fast'
        if not os.path.isdir(args.o):
            os.mkdir(args.o)
    if args.i is None:
        args.i = 'metabs_toxin'
    if args.type is None:
        args.type = 'auc'
    if args.week is None:
        args.week = 0
    else:
        args.week = [float(w) for w in args.week.split('_')]
    if args.folds is None:
        args.folds = 0
    if args.num_folds is None and args.folds == 1:
        args.num_folds = 5

    if isinstance(args.week, list):
        if len(args.week)==1:
            args.week = args.week[0]

    dl = dataLoader(pt_perc={'metabs': .25, '16s': .1, 'scfa': 0}, meas_thresh=
            {'metabs': 0, '16s': 10, 'scfa': 0}, var_perc={'metabs': 25, '16s': 5, 'scfa': 0})

    if isinstance(args.week, list):
        x, y, event_times = get_slope_data(dl, args.i, args.week)
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
        train_index, test_index = ixs[args.ix]
        x_train0, x_test0 = x.iloc[train_index, :], x.iloc[test_index, :]
        y_train0, y_test0 = y[train_index], y[test_index]

        if args.folds == 1:
            final_res_dict = train_lr_folds(x_train0, y_train0, num_folds=args.num_folds)
        else:
            final_res_dict = train_lr(x_train0, y_train0, path_out = path_out)
    elif args.type == 'coef':
        if args.folds == 1:
            final_res_dict = train_lr_folds(x,y, num_folds=args.num_folds)
        else:
            final_res_dict = train_lr(x,y, path_out = path_out)

    final_res_dict['data'] = (x, y)

    with open(path_out + args.type + '_ix_' + str(args.ix)+ '.pkl', 'wb') as f:
        pkl.dump(final_res_dict, f)

    end = time.time()
    passed = np.round((end - start) / 60, 3)
    f2 = open(args.o + '/' + args.i + ".txt", "a")
    try:
        f2.write('index ' + str(args.ix) + ', CI ' + str(final_res_dict['score']) +' in ' + str(passed) + ' minutes' + '\n')
    except:
        f2.write(
            'index ' + str(args.ix) + ' in ' + str(passed) + ' minutes' + '\n')
    f2.close()
