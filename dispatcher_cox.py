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
#BSUB -m bwhpath_h

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

python3 ./main_cox.py -ix {0} {1} -o {2} -i {3} -metric {4}
'''
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--o", help = "outpath", type = str)
args = parser.parse_args()

metric = 'CI'

if not args.o:
    print('Specify out dir')
    sys.exit(1)

if not os.path.isdir(args.o):
    os.mkdir(args.o)

dattype = 'metabs'

for ix in range(49):
    for ix1 in range(48):
        path_out = args.o + '/' + dattype + '/'
        if os.path.exists(path_out + 'ix_' + str(ix) +'ix_' + str(ix1)+ '.pkl'):
            continue
        else:
            fname = 'cdiff_lr.lsf'
            f = open(fname, 'w')
            f.write(my_str.format(ix, ix1, args.o, dattype, metric))
            f.close()
            os.system('bsub < {}'.format(fname))




