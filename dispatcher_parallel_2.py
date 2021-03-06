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
#BSUB -J pylab
#BSUB -o fl.out
#BSUB -e fl.err

# This is a sample script with specific resource requirements for the
# **bigmemory** queue with 64GB memory requirement and memory
# limit settings, which are both needed for reservations of
# more than 40GB.
# Copy this script and then submit job as follows:
# ---
# cd ~/lsf
# cp templates/bsub/example_8CPU_bigmulti_64GB.lsf .
# bsub < example_bigmulti_8CPU_64GB.lsf
# ---
# Then look in the ~/lsf/output folder for the script log
# that matches the job ID number

# Please make a copy of this script for your own modifications

#BSUB -q rerunnable
#BSUB -n 12
#BSUB -M 10000
#BSUB -R rusage[mem=10000]

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
            fname = 'cdiff_lr.lsf'
            f = open(fname, 'w')
            f.write(my_str.format(seed, param, ic, out_path, input_path, model, feat[0], feat[1],False))
            f.close()
            os.system('bsub < {}'.format(fname))
    time.sleep(0.5)

time.sleep(5)
for seed in range(50):
    for ic in range(48):
        f.write(my_str.format(seed,param,ic,out_path,input_path,model,0,0,True))
    # else:
    #     ic = 0
    #     fname = 'cdiff_lr.lsf'
    #     f = open(fname, 'w')
    #     f.write(my_str.format(seed, param, ic, out_path, input_path, model))
    #     f.close()
    #     os.system('bsub < {}'.format(fname))


