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
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
warnings.filterwarnings("ignore")

def train_cox(x, split_to_folds = leave_two_out):
    # if feature_grid is None:
    #     feature_grid = np.logspace(7, 20, 14)
    hazards = []
    event_times = []
    event_outcomes = []
    score_vec = []
    model_out_dict = {}
    ix_inner = split_to_folds(x, x['outcome'])
    lambda_dict = {}
    for ic_in, ix_in in enumerate(ix_inner):
        train_index, test_index = ix_in
        x_train, x_test = x.iloc[train_index, :], x.iloc[test_index, :]

        ix_inner2 = split_to_folds(x_train, x_train['outcome'])
        lamb_dict = {}
        lamb_dict['auc'] = {}
        lamb_dict['ci'] = {}
        model2 = CoxnetSurvivalAnalysis(l1_ratio=1, alpha_min_ratio=.0001, n_alphas=100)

        model_dict = {}
        alphas = None
        hazards_dict = {}
        e_times_dict = {}
        e_outcomes_dict = {}
        score_dict = {}
        for ic_in2, ix_in2 in enumerate(ix_inner2):
            start_inner = time.time()

            train_ix, test_ix = ix_in2
            x_tr2, x_ts2 = x_train.iloc[train_ix, :], x_train.iloc[test_ix, :]
            week = x_tr2['week']
            outcome = x_tr2['outcome']
            if (outcome == 0).all():
                continue
            x_tr2_ = x_tr2.drop(['week', 'outcome'], axis=1)
            yy = list(zip(outcome, week))
            y_arr = np.array(yy, dtype=[('e.tdm', '?'), ('t.tdm', '<f8')])
            model2.set_params(alphas=alphas)
            model2.fit(x_tr2_, y_arr)
            alphas_new = model2.alphas_
            if ic_in2 == 0:
                alphas = alphas_new

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
                    score_dict[i][ic_in2], _, _,_,_ = concordance_index_censored(e_outcomes_dict[i][ic_in2].astype(bool), e_times_dict[i][ic_in2],
                                                                   hazards_dict[i][ic_in2])

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
        week = x_train['week']
        outcome = x_train['outcome']
        x_train_ = x_train.drop(['week','outcome'], axis = 1)
        yy = list(zip(outcome, week))
        y_arr = np.array(yy, dtype = [('e.tdm', '?'), ('t.tdm', '<f8')])
        model_out.fit(x_train_, y_arr)

        risk_scores = model_out.predict(x_test.drop(['week', 'outcome'], axis=1), alpha = best_lamb)

        hazards.append(risk_scores)
        event_times.append(x_test['week'])
        event_outcomes.append(x_test['outcome'])

        model_out_dict[ic_in] = model_out
        if len(test_index) > 1:
            score_vec.append(concordance_index_censored(x_test['outcome'].astype(bool), x_test['week'], risk_scores)[0])

    if len(test_index) > 1:
        score = sum(score_vec)/len(score_vec)
    else:
        score, concordant, discondordant, tied_risk, tied_time = concordance_index_censored(
            np.array(np.concatenate(event_outcomes)).astype(bool),
            np.array(np.concatenate(event_times)), np.array(np.concatenate(hazards)))

    final_dict = {}
    final_dict['ci'] = score
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
    args = parser.parse_args()

    if args.ix is None:
        args.ix = 0
        args.o = 'test_cox_preds_fast'
        args.i = 'metabs'
        args.type = 'auc'
        args.week = 1
    else:
        args.week = [float(w) for w in args.week.split('_')]

    if args.seed is None:
        args.seed = 0

    if isinstance(args.week, list):
        if len(args.week)==1:
            args.week = args.week[0]

    if args.i == '16s':
        dl = dataLoader(pt_perc=.05, meas_thresh=10, var_perc=5, pt_tmpts=1)
    else:
        dl = dataLoader(pt_perc=.25, meas_thresh=0, var_perc=15, pt_tmpts=1)

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

        final_res_dict = train_cox(x_train0)
    elif args.type == 'coef':
        final_res_dict = train_cox(x)

    final_res_dict['data'] = x

    with open(path_out + args.type + '_ix_' + str(args.ix)+ '.pkl', 'wb') as f:
        pkl.dump(final_res_dict, f)

    end = time.time()
    passed = np.round((end - start) / 60, 3)
    f2 = open(args.o + '/' + args.i + ".txt", "a")
    f2.write('index ' + str(args.ix) + ', CI ' + str(final_res_dict['ci']) +' in ' + str(passed) + ' minutes' + '\n')
    f2.close()

# Get coefficients from nested CV
