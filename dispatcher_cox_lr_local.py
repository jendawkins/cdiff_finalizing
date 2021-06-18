import numpy as np
import os
import shutil
import argparse
import pickle
import time
import sys
import itertools
import subprocess

my_str_cox = '''python ./main_cox_fast.py -ix {0} -i {1} -o {2} -type {3} -week {4}'''

my_str_lr = '''python ./main_parallel.py -ix {0} -i {1} -o {2} -type {3} -week {4}'''

parser = argparse.ArgumentParser()
parser.add_argument("-o","--o",help = 'out file', type = str)
parser.add_argument("-model","--model",help = 'model', type = str)
parser.add_argument("-week","--week",help = 'week', type = float, nargs = '+')
args = parser.parse_args()

if args.model == 'cox':
    my_str = my_str_cox
elif args.model == 'LR':
    my_str = my_str_lr

if len(args.week) > 1:
    args.week = '_'.join([str(w) for w in args.week])
    args.week = args.week.replace('.','a')

out_path = 'FinalRuns/' + args.model + '_week' + str(args.week)
if not os.path.isdir(out_path):
    os.mkdir(out_path)
if not args.o:
    args.o = out_path

inputs = 'metabs'
max_load = 10
pid_list = []

cmnd = my_str.format(0, inputs, args.o, 'coef', args.week)
args2 = cmnd.split(' ')
print(args2)
pid = subprocess.Popen(args2)
pid_list.append(pid)

for ix in np.arange(49):
    cmnd = my_str.format(ix, inputs, args.o, 'auc', args.week)
    args2 = cmnd.split(' ')
    print(args2)
    pid = subprocess.Popen(args2)
    pid_list.append(pid)
    while sum([x.poll() is None for x in pid_list]) >= max_load:
        time.sleep(30)
