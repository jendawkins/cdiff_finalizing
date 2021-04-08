import numpy as np
import os
import shutil
import argparse
import pickle
import time
import sys
import itertools

my_str = '''
python3 ./main_parallel_2.py -seed {0} -param {1} -ix {2} -o {3} -i {4} -model {5} -test_feat {6} {7} -final {8}
'''
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--o", help = "outpath", type = str)
args = parser.parse_args()
out_path = args.o

if not args.o:
    print('Specify out dir')
    sys.exit(1)

if not os.path.isdir(out_path):
    os.mkdir(out_path)

model = 'RF'
param = 'auc'
input_path = 'week_one_metabs'

if model == 'LR':
    feature_grid = np.logspace(-3,3,100)
else:
    estimators_grid = np.arange(2,51,2)
    depth_grid = np.arange(2,20,1)
    feature_grid = list(itertools.product(estimators_grid, depth_grid))

for feat in feature_grid:
    if isinstance(feat, tuple):
        feat = list(feat)
    else:
        feat = [feat]
    for ic in range(48):
        for seed in range(50):
            cmnd = my_str.format(seed, param, ic, out_path, input_path, model, feat[0], feat[1],0)
            os.system(cmnd)

time.sleep(5)
for seed in range(50):
    for ic in range(48):
        f.write(my_str.format(seed,param,ic,out_path,input_path,model,0,0,1))
    # else:
    #     ic = 0
    #     fname = 'cdiff_lr.lsf'
    #     f = open(fname, 'w')
    #     f.write(my_str.format(seed, param, ic, out_path, input_path, model))
    #     f.close()
    #     os.system('bsub < {}'.format(fname))


