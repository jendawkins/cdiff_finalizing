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

python3 ./main_parallel.py -seed {0} -param {1} -ix {2} -o {3} -i {4}
'''
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--o", help = "outpath", type = str)
args = parser.parse_args()

if not args.o:
    print('Specify out dir')
    sys.exit(1)

if not os.path.isdir(args.o):
    os.mkdir(args.o)

model = 'LR'
param = 'auc_bootstrap'

for seed in range(0,50):
    # for model in ['LR','RF']:
    for input_path in ['metabs','16s','bile_acids','joint']:
    # for input_path in ['week_one_ALL', 'week_one_metabs','week_one_16s','week_one_bileacids']:
        for ix in range(49):
            path_out = args.o + '/' + input_path + '/'
            if os.path.exists(path_out + "_" + param + "_" + str(seed) + ".pkl"):
                continue
            else:
                fname = 'cdiff_lr.lsf'
                f = open(fname, 'w')
                f.write(my_str.format(seed, param, ix, args.o, input_path, model))
                f.close()
                os.system('bsub < {}'.format(fname))



