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
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
warnings.filterwarnings("ignore")


def train_cox(x, outer_split = leave_two_out, inner_split = leave_two_out, num_folds = None):
    if num_folds is None:
        print('none')
    else:
        print(num_folds)
    # if feature_grid is None:
    #     feature_grid = np.logspace(7, 20, 14)
    hazards = []
    event_times = []
    event_outcomes = []
    score_vec = []
    model_out_dict = {}
    ix_inner = outer_split(x, x['outcome'], num_folds=None)
    lambda_dict = {}
    for ic_in, ix_in in enumerate(ix_inner):
        train_index, test_index = ix_in
        x_train, x_test = x.iloc[train_index, :], x.iloc[test_index, :]

        if np.sum(x_test['outcome'].values)<1:
            continue

        week = x_train['week']
        outcome = x_train['outcome']
        x_train_ = x_train.drop(['week','outcome'], axis = 1)
        yy = list(zip(outcome, week))
        y_arr = np.array(yy, dtype = [('e.tdm', '?'), ('t.tdm', '<f8')])

        ix_inner2 = inner_split(x_train, x_train['outcome'], num_folds = None)
        lamb_dict = {}
        lamb_dict['auc'] = {}
        lamb_dict['ci'] = {}
        model2 = CoxnetSurvivalAnalysis(l1_ratio=1)

        model_dict = {}
        alphas = None
        hazards_dict = {}
        e_times_dict = {}
        e_outcomes_dict = {}
        score_dict = {}

        coxnet_pipe = CoxnetSurvivalAnalysis(l1_ratio=1, alpha_min_ratio=0.001,n_alphas=300)

        coxnet_pipe.fit(x_train_, y_arr)
        alphas = coxnet_pipe.alphas_

        for ic_in2, ix_in2 in enumerate(ix_inner2):
            start_inner = time.time()

            train_ix, test_ix = ix_in2
            x_tr2, x_ts2 = x_train.iloc[train_ix, :], x_train.iloc[test_ix, :]

            if np.sum(x_tr2['outcome'].values) < 1:
                continue

            y_test = list(zip(x_ts2['outcome'], x_ts2['week']))
            y_test_arr = np.array(y_test, dtype=[('e.tdm', '?'), ('t.tdm', '<f8')])
            if len(np.unique(y_test_arr)) < len(test_ix):
                continue

            week = x_tr2['week']
            outcome = x_tr2['outcome']
            if (outcome == 0).all():
                continue
            x_tr2_ = x_tr2.drop(['week', 'outcome'], axis=1)
            yy2 = list(zip(outcome, week))
            y_arr2 = np.array(yy2, dtype=[('e.tdm', '?'), ('t.tdm', '<f8')])
            model2.set_params(alphas=alphas)
            try:
                model2.fit(x_tr2_, y_arr2)
            except:
                print('removed alpha ' + str(alphas[0]))
                alphas_n = np.delete(alphas, 0)
                model2.set_params(alphas=alphas_n)
                while(1):
                    try:
                        model2.fit(x_tr2_, y_arr2)
                        alphas = alphas_n
                        break
                    except:
                        print('removed alpha ' + str(alphas_n[0]))
                        alphas_n = np.delete(alphas, 0)
                        model2.set_params(alphas=alphas_n)
                    if len(alphas_n)<=2:
                        break
                if len(alphas_n)<=2:
                    continue
            # alphas_new = model2.alphas_
            # if ic_in2 == 0:
            #     alphas = alphas_new

            model_dict[ic_in2] = model2
            for i, alpha in enumerate(alphas):
                if i not in hazards_dict.keys():
                    hazards_dict[i]={}
                    e_times_dict[i] = {}
                    e_outcomes_dict[i] = {}
                    score_dict[i] = {}
                risk_scores = model2.predict(x_ts2.drop(['week', 'outcome'], axis=1), alpha=alpha)
                hazards_dict[i][ic_in2] = risk_scores
                e_times_dict[i][ic_in2] = x_ts2['week']
                e_outcomes_dict[i][ic_in2] = x_ts2['outcome']

                if len(test_ix)>=2:
                    ci = concordance_index_censored(e_outcomes_dict[i][ic_in2].astype(bool), e_times_dict[i][ic_in2],
                                                                       hazards_dict[i][ic_in2])
                    if not np.isnan(ci):
                        score_dict[i][ic_in2], _, _,_,_ = ci


        if len(score_dict[i]) > 0:
            scores = {i: sum(score_dict[i].values())/len(score_dict[i].values()) for i in score_dict.keys()}
        else:
            scores = {}
            for a_ix in hazards_dict.keys():
                alpha_num = alphas[a_ix]
                scores[alpha_num], concordant, discondordant, tied_risk, tied_time = concordance_index_censored(
                    np.array(np.concatenate(list(e_outcomes_dict[a_ix].values()))).astype(bool),
                    np.array(np.concatenate(list(e_times_dict[a_ix].values()))),
                    np.array(np.concatenate(list(hazards_dict[a_ix].values()))))

        lambdas, aucs_in = list(zip(*scores.items()))
        ix_max = np.argmax(aucs_in)
        best_lamb = lambdas[ix_max]

        lambda_dict[ic_in] = {'best_lambda': best_lamb, 'scores': scores, 'event_outcomes':event_outcomes, 'times':event_times,
                       'hazards':hazards, 'lambdas_tested': alphas}
        model_out = CoxnetSurvivalAnalysis(l1_ratio = 1, alphas = alphas)

        model_out.fit(x_train_, y_arr)

        risk_scores = model_out.predict(x_test.drop(['week', 'outcome'], axis=1), alpha = best_lamb)

        hazards.append(risk_scores)
        event_times.append(x_test['week'])
        event_outcomes.append(x_test['outcome'])

        model_out_dict[ic_in] = model_out
        if len(test_index) > 1:
            ci = concordance_index_censored(x_test['outcome'].astype(bool), x_test['week'], risk_scores)[0]
            if not np.isnan(ci):
                score_vec.append(ci)

    if len(score_vec) > 1:
        score = sum(score_vec)/len(score_vec)
    else:
        score, concordant, discondordant, tied_risk, tied_time = concordance_index_censored(
            np.array(np.concatenate(event_outcomes)).astype(bool),
            np.array(np.concatenate(event_times)), np.array(np.concatenate(hazards)))

    final_dict = {}
    final_dict['score'] = score
    final_dict['model'] = model_out_dict
    final_dict['hazards'] = hazards
    final_dict['event_times'] = event_times
    final_dict['event_outcomes'] = event_outcomes
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
    parser.add_argument("-seed", "--seed", help="seed", type=int)
    parser.add_argument("-folds", "--folds", help="use_folds", type=int)
    parser.add_argument("-num_folds", "--num_folds", help="num_folds", type=int)
    args = parser.parse_args()

    if args.ix is None:
        args.ix = 0
    if args.o is None:
        args.o = 'test_lr_fast'
        if not os.path.isdir(args.o):
            os.mkdir(args.o)
    if args.i is None:
        args.i = '16s'
    if args.type is None:
        args.type = 'auc'
    if args.week is None:
        args.week = 1
    else:
        args.week = [float(w) for w in args.week.split('_')]
    if args.folds is None:
        args.folds = 0
    if args.num_folds is None:
        args.num_folds = None

    np.random.seed(args.seed)
    if isinstance(args.week, list):
        if len(args.week)==1:
            args.week = args.week[0]

    dl = dataLoader(pt_perc={'metabs': .25, '16s': .1, 'scfa': 0}, meas_thresh=
        {'metabs': 0, '16s': 10, 'scfa': 0}, var_perc={'metabs': 15, '16s': 5, 'scfa': 0})
    if isinstance(args.week, list):
        x, outcomes, event_times = get_slope_data(dl.week[args.i], args.week)
    else:
        data = dl.week[args.i][args.week]
        x, outcomes, event_times = data['x'], data['y'], data['event_times']
        x.index = [xind.split('-')[0] for xind in x.index.values]
    x['week'] = event_times
    x['outcome'] = (np.array(outcomes) == 'Recurrer').astype(float)

    path_out = args.o + '/' + args.i + '/'

    if not os.path.isdir(path_out):
        os.mkdir(path_out)

    np.random.seed(args.seed)
    if args.type == 'auc':
        ixs = leave_one_out_cv(x, x['outcome'])
        train_index, test_index = ixs[args.ix]
        x_train0, x_test0 = x.iloc[train_index, :], x.iloc[test_index, :]

        final_res_dict = train_cox(x_train0, num_folds=args.num_folds)
            # final_res_dict = train_cox(x_train0)
    elif args.type == 'coef':
        final_res_dict = train_cox(x, num_folds=args.num_folds)
            # final_res_dict = train_cox(x)

    final_res_dict['data'] = x

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

# Get coefficients from nested CV
