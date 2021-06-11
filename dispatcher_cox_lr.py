import numpy as np
import os
import shutil
import argparse
import pickle
import time
import sys
import itertools

my_str = '''
#!/bin/bash
#BSUB -J test
#BSUB -o output/test-%J.out
#BSUB -e output/test-%J.err

# This is a sample script with specific resource requirements
# for the **normal** general queue with modest memory requirement 
# 8GB or less memory, default memory allocation 2GB. 
# Maximum runtime 3 day. Maximum number of CPU cores 6.
# Copy this script and then submit job as follows:
# ---
# cd ~/lsf
# cp templates/bsub/example_normal_6CPU_8GB.lsf .
# bsub < example_normal_6CPU_8GB.lsf
# ---
# Then look in the ~/lsf/output folder for the script log
# that matches the job ID number

# Please make a copy of this script for your own modifications

#BSUB -q rerunnable

# Some important variables to check (Can be removed later)
echo '---PROCESS RESOURCE LIMITS---'
ulimit -a
echo '---SHARED LIBRARY PATH---'
echo $LD_LIBRARY_PATH
echo '---APPLICATION SEARCH PATH:---'
echo $PATH
echo '---LSF Parameters:---'
printenv | grep '^LSF'
echo '---LSB Parameters:---'
printenv | grep '^LSB'
echo '---LOADED MODULES:---'
module list
echo '---SHELL:---'
echo $SHELL
echo '---HOSTNAME:---'
hostname
echo '---GROUP MEMBERSHIP (files are created in the first group listed):---'
groups
echo '---DEFAULT FILE PERMISSIONS (UMASK):---'
umask
echo '---CURRENT WORKING DIRECTORY:---'
pwd
echo '---DISK SPACE QUOTA---'
df .
echo '---TEMPORARY SCRATCH FOLDER ($TMPDIR):---'
echo $TMPDIR

# Add your job command here

cd /PHShome/jjd65/cdiff_finalizing
'''

my_str_cox = '''python ./main_cox_fast.py -ix {0} -i {1} -o {2} -type {3} -week {4}'''

my_str_lr = '''python ./main_parallel.py -ix {0} -i {1} -o {2} -type {3} -week {4}'''

parser = argparse.ArgumentParser()
parser.add_argument("-o","--o",help = 'out file', type = str)
parser.add_argument("-model","--model",help = 'model', type = str)
parser.add_argument("-week","--week",help = 'week', type = float)
args = parser.parse_args()

if args.model == 'cox':
    my_str = my_str + my_str_cox
elif args.model == 'LR':
    my_str = my_str + my_str_lr

if not os.path.isdir('FinalRuns'):
    os.mkdir('FinalRuns')
out_path = 'FinalRuns/' + args.model + '_week' + str(args.week)
if not os.path.isdir(out_path):
    os.mkdir(out_path)
if not args.o:
    args.o = out_path

dattype = 'metabs'

fname = 'cdiff_lr.lsf'
f = open(fname, 'w')
f.write(my_str.format(0, dattype, args.o, 'coef', args.week))
f.close()
os.system('bsub < {}'.format(fname))

for ix in range(49):
    path_out = args.o + '/' + dattype + '/'
    if os.path.exists(path_out + "_" + 'auc'+ "_" + str(0) + "_" + str(ix) + '.pkl','wb'):
        continue
    else:
        fname = 'cdiff_lr.lsf'
        f = open(fname, 'w')
        f.write(my_str.format(ix, dattype, args.o, 'auc', args.week))
        f.close()
        os.system('bsub < {}'.format(fname))




