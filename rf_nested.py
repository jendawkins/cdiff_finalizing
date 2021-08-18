import argparse
import os
import time
import pickle as pkl
from sklearn.ensemble import RandomForestClassifier
import time
from helper import *
from dataLoader import *
from basic_data_methods import *
import itertools

def run_rf(x,y,random_state = 0, n_estimators=[50,100], max_depth = None,
           max_features = [None,'auto'], min_samples_split = [2,9], min_samples_leaf = [1,5]):
    probs = []
    outcomes = []
    model_out_dict = {}
    ix_inner = leave_one_out_cv(x,y)
    lambda_dict = {}

    for ic_in, ix_in in enumerate(ix_inner):
        train_index, test_index = ix_in
        x_train, x_test = x.iloc[train_index, :], x.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        ix_inner2 = leave_one_out_cv(x_train, y_train)

        model_dict = {}
        test_probs = {}
        y_true = {}

        for ic_in2, ix_in2 in enumerate(ix_inner2):

            train_ix, test_ix = ix_in2
            x_tr2, x_ts2 = x_train.iloc[train_ix, :], x_train.iloc[test_ix, :]
            y_tr2, y_ts2 = y_train[train_ix], y_train[test_ix]

            if sum(y_tr2)==0:
                continue

            params = list(itertools.product(n_estimators, max_features, min_samples_split, min_samples_leaf))
            plist = []
            for n_est, max_feat, min_samp_split, min_samp_leaf in params:
                rf = RandomForestClassifier(random_state=random_state, class_weight='balanced', bootstrap=True,
                                            max_features=max_feat, oob_score=True, min_samples_split=min_samp_split,
                                            min_samples_leaf=min_samp_leaf, n_estimators=n_est, max_depth = max_depth)
                rf.fit(x_tr2, y_tr2)

                param_key = (('n_est', n_est), ('max_feat', max_feat), ('min_samp_split', min_samp_split),
                             ('min_samp_leaf', min_samp_leaf))
                if param_key not in test_probs.keys():
                    test_probs[param_key]={}
                    model_dict[param_key]={}
                test_probs[param_key][ic_in2]= rf.predict_proba(x_ts2)
                model_dict[param_key][ic_in2] = rf
                plist.append(param_key)
            y_true[ic_in2] = y_ts2
        scores = {}
        pt_ixs = list(y_true.keys())
        for l_ix in test_probs.keys():
            scores[l_ix] = sklearn.metrics.roc_auc_score([y_true[iix].item() for iix in pt_ixs],
                                                         [test_probs[l_ix][iix][:,1].item() for iix in pt_ixs])
        lambdas, aucs_in = list(zip(*scores.items()))
        ix_max = np.argmax(aucs_in)
        best_lamb = lambdas[ix_max]

        lambda_dict[ic_in] = {'best_lambda': best_lamb, 'scores': scores, 'outcomes':y_true,
                       'probs':test_probs, 'lambdas_tested': plist}

        best_out = dict(best_lamb)
        model_out = RandomForestClassifier(random_state = random_state, class_weight = 'balanced', bootstrap=True,
                                            max_features=best_out['max_feat'], oob_score=True,
                                           min_samples_split=best_out['min_samp_split'], max_depth = max_depth,
                                            min_samples_leaf=best_out['min_samp_leaf'], n_estimators=best_out['n_est'])
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
    parser.add_argument("-random_state", "--random_state", help="random state", type=int)
    args = parser.parse_args()

    if args.ix is None:
        args.ix = 0
    if args.o is None:
        args.o = 'test_rf_fast'
        if not os.path.isdir(args.o):
            os.mkdir(args.o)
    if args.i is None:
        args.i = 'metabs'
    if args.type is None:
        args.type = 'auc'
    if args.week is None:
        args.week = [1,1.5,2]
    else:
        args.week = [float(w) for w in args.week.split('_')]
    if args.folds is None:
        args.folds = 0
    if args.num_folds is None and args.folds == 1:
        args.num_folds = 5

    if isinstance(args.week, list):
        if len(args.week)==1:
            args.week = args.week[0]

    dl = dataLoader(pt_perc={'metabs': .25, '16s': .1, 'scfa': 0, 'toxin':0}, meas_thresh=
            {'metabs': 0, '16s': 10, 'scfa': 0, 'toxin':0}, var_perc={'metabs': 15, '16s': 5, 'scfa': 0, 'toxin':0})

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

        final_res_dict = run_rf(x_train0, y_train0)
    elif args.type == 'coef':
         final_res_dict = run_rf(x,y)

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