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
import argparse


from basic_ml import *

mb = basic_ml()

parser = argparse.ArgumentParser()
parser.add_argument("-seed", "--seed", help="random_seed", type=int)

args = parser.parse_args()

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

final_res_dict = {}

path_in = 'week_one_bileacids'
x,y = dat_dict[path_in]

start = time.time()
model = LogisticRegression(class_weight = 'balanced', penalty = 'l1', \
                            random_state = args.seed, solver = 'liblinear')
final_res_dict = mb.nested_cv_func(model, x, y, dtype = None, optim_param = 'auc',\
                                        plot_lambdas = False, learn_var = 'C',
                                    feature_grid = np.logspace(-9,3,100), smooth_auc = True)

with open('outputs_ba_local/' + str(time.time()).replace('.','_') + '.pkl','wb') as f:
    pickle.dump(final_res_dict, f)