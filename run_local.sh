#!/usr/bin/env bash
python3 ./main_parallel.py -seed 0 -param 'auc' -o 'out_test' -i 'week_one_16s'
python3 ./main_parallel.py -seed 0 -param 'auc' -o 'out_test' -i 'all_data_16s'
python3 ./main_parallel.py -seed 0 -param 'auc' -o 'out_test' -i 'all_data'
python3 ./main_parallel.py -seed 0 -param 'auc' -o 'out_test' -i 'week_one'

python3 ./main_parallel.py -seed 1 -param 'auc' -o 'out_test' -i 'week_one_16s'
python3 ./main_parallel.py -seed 1 -param 'auc' -o 'out_test' -i 'all_data_16s'
python3 ./main_parallel.py -seed 1 -param 'auc' -o 'out_test' -i 'all_data'
python3 ./main_parallel.py -seed 1 -param 'auc' -o 'out_test' -i 'week_one'

python3 ./main_parallel.py -seed 2 -param 'auc' -o 'out_test' -i 'week_one_16s'
python3 ./main_parallel.py -seed 2 -param 'auc' -o 'out_test' -i 'all_data_16s'
python3 ./main_parallel.py -seed 2 -param 'auc' -o 'out_test' -i 'all_data'
python3 ./main_parallel.py -seed 2 -param 'auc' -o 'out_test' -i 'week_one'

python3 ./main_parallel.py -seed 3 -param 'auc' -o 'out_test' -i 'week_one_16s'
python3 ./main_parallel.py -seed 3 -param 'auc' -o 'out_test' -i 'all_data_16s'
python3 ./main_parallel.py -seed 3 -param 'auc' -o 'out_test' -i 'all_data'
python3 ./main_parallel.py -seed 3 -param 'auc' -o 'out_test' -i 'week_one'

python3 ./main_parallel.py -seed 4 -param 'auc' -o 'out_test' -i 'week_one_16s'
python3 ./main_parallel.py -seed 4 -param 'auc' -o 'out_test' -i 'all_data_16s'
python3 ./main_parallel.py -seed 4 -param 'auc' -o 'out_test' -i 'all_data'
python3 ./main_parallel.py -seed 4 -param 'auc' -o 'out_test' -i 'week_one'

python3 ./main_parallel.py -seed 5 -param 'auc' -o 'out_test' -i 'week_one_16s'
python3 ./main_parallel.py -seed 5 -param 'auc' -o 'out_test' -i 'all_data_16s'
python3 ./main_parallel.py -seed 5 -param 'auc' -o 'out_test' -i 'all_data_16s'
python3 ./main_parallel.py -seed 5 -param 'auc' -o 'out_test' -i 'week_one'