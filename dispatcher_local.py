import numpy as np
import os
import shutil
import argparse
import pickle
import time
import sys
import itertools
import subprocess

my_str = '''
python3 ./main_parallel.py -seed {0} -param {1} -ix {2} -o {3} -i {4} -model {5} 
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

for input_path in ['week_one_metabs', 'week_one_16s']:
    for seed in range(10):
        start = time.time()
        cmnd = my_str.format(seed, param, 0, out_path, input_path, model)
        os.system(cmnd)
        end = time.time() - start
        print(input_path + ' seed ' + str(seed)+ ': ' +str(end))

