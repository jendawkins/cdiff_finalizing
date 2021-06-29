import argparse
import os
import time
import pickle as pkl
from sklearn.ensemble import RandomForestClassifier
import time
from helper import *
from dataLoader import *
from basic_data_methods import *

def nullable_string(val):
    if not val:
        return None
    return val

def argparse_to_str(args):
    str_ls = [arg + '_' + str(getattr(args, arg)) for arg in vars(args)]
    return '__'.join(str_ls)

parser = argparse.ArgumentParser()
parser.add_argument("-random_state","--random_state",type = int)
parser.add_argument("-week","--week", type = str)
# parser.add_argument("-class_weight","--class_weight",type = str)
parser.add_argument("-n_estimators","--n_estimators",type = int)
parser.add_argument("-min_samples_split","--min_samples_split",type = int)
# parser.add_argument("-bootstrap","--bootstrap",type = int)
parser.add_argument("-max_features","--max_features",type = nullable_string)
parser.add_argument("-max_depth","--max_depth",type = nullable_string)
parser.add_argument("-min_samples_leaf","--min_samples_leaf",type = int)
parser.add_argument("-use_univariate","--use_univariate",type = int)
parser.add_argument("-o", "--o", help="outpath", type=str)
parser.add_argument("-i","--i",type = str)
parser.add_argument("-folds","--folds",type = int)
parser.add_argument("-num_folds","--num_folds",type = int)
parser.add_argument("-type","--type",type = str)
args = parser.parse_args()

if args.i is None:
    args.i = 'metabs'
if args.week is None:
    args.week = 1
elif isinstance(args.week, list) or isinstance(args.week, str):
    args.week = [float(w) for w in args.week.split('_')]
    if len(args.week)==1:
        args.week = args.week[0]
elif isinstance(args.week, int) or isinstance(args.week, float):
    args.week = float(args.week)

if args.max_depth:
    args.max_depth = int(args.max_depth)

if args.random_state is None:
    args.random_state = 0
if args.n_estimators is None:
    args.n_estimators = 100
if args.min_samples_split is None:
    args.min_samples_split = 9
if args.max_features is None:
    args.max_features = None
if args.min_samples_leaf is None:
    args.min_samples_leaf = 1
if args.use_univariate is None:
    args.use_univariate = 0
if args.folds is None:
    args.folds = 0
if args.num_folds is None:
    args.num_folds = 5

if not os.path.isdir(args.o):
    os.mkdir(args.o)

path_out = args.o + '/' + args.i + '/'

if not os.path.isdir(path_out):
    os.mkdir(path_out)


rf = RandomForestClassifier(random_state = args.random_state, class_weight = 'balanced',\
                            n_estimators = args.n_estimators, min_samples_split = args.min_samples_split, \
                            bootstrap = True, max_features = args.max_features,
                            oob_score = 1, max_depth = args.max_depth, min_samples_leaf=args.min_samples_leaf)


dl = dataLoader(pt_perc={'metabs': .25, '16s': .05, 'scfa': 0}, meas_thresh=
    {'metabs': 0, '16s': 10, 'scfa': 0}, var_perc={'metabs': 15, '16s': 5, 'scfa': 0})
if isinstance(args.week, list):
    x, y, event_times = get_slope_data(dl.week[args.i], args.week)
else:
    data = dl.week[args.i][args.week]
    x, y, event_times = data['x'], data['y'], data['event_times']
    x.index = [xind.split('-')[0] for xind in x.index.values]

try:
    y = (np.array(y)=='Recurrer').astype('float')
except:
    print(args.week)

ixs = leave_one_out_cv(x,y)

probs = []
true = []
preds = []
params = []
aucs = []
importances = []
clf_out_dict = {}
clf_dict = {}
res_dict = {}
auc_inner = {}
for i, ix in enumerate(ixs):
    tr, ts = ix
    x_tr, x_ts = x.iloc[tr, :], x.iloc[ts, :]
    y_tr, y_ts = y[tr], y[ts]

    if args.folds == 0:
        ixs_tr = leave_one_out_cv(x_tr, y_tr)
    else:
        num_rec = np.sum(y)
        if args.num_folds > (num_rec-1):
            args.num_folds = num_rec-1
        skf = StratifiedKFold(n_splits=args.num_folds)
        ixs_tr = skf.split(x_tr, y_tr)
        auc_inner[i]={}

    clf_dict[i] = {}
    for ii, ix_in in enumerate(ixs_tr):
        trr, tss = ix_in
        x_trr, x_tss = x_tr.iloc[trr, :], x_tr.iloc[tss, :]
        y_trr, y_tss = y_tr[trr], y_tr[tss]

        if args.use_univariate == 1:
            df = rank_sum(x_trr, y_trr)
            feats = df.index.values[df['P_Val'] <= np.percentile(df['P_Val'],1)]
            clf = rf.fit(x_trr[feats], y_trr)
            pred_probs = clf.predict_proba(x_tss[feats])
            pred = clf.predict(x_tss[feats])
        else:
            clf = rf.fit(x_trr, y_trr)
            pred_probs = clf.predict_proba(x_tss)
            pred = clf.predict(x_tss)

        if args.folds == 0:
            probs.append(pred_probs.squeeze()[1])
            true.append(y_tss)
            preds.append(pred)
        else:
            auc_inner[i][ii] = sklearn.metrics.roc_auc_score(y_tss, pred_probs[:,1])
        clf_dict[i][ii] = clf

    if args.use_univariate == 1:
        clf_out = rf.fit(x_tr[feats], y_tr)
    else:
        clf_out = rf.fit(x_tr, y_tr)

    clf_out_dict[ts[0]] = clf_out
    importances.append(clf_out.feature_importances_)
    if args.folds == 0:
        auc_score = sklearn.metrics.roc_auc_score(true, probs)
        aucs.append(auc_score)
if args.folds == 1:
    aucs = auc_inner
res_dict['score'] = aucs
res_dict['importances'] = importances
res_dict['data'] = x

with open(path_out + args.type + '_ix_' + str(args.random_state) + '.pkl', 'wb') as f:
    pkl.dump(res_dict, f)

f2 = open(args.o + '/' + args.i + ".txt", "a")
try:
    f2.write(
        'state ' + str(args.random_state) + ', score ' + str(res_dict['score']) + '\n')
except:
    f2.write(
        'state ' + str(args.random_args.state) + '\n')
f2.close()
