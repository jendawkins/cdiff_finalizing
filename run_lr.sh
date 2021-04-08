#!/bin/bash
for i in {1..50}; do
  echo -e "\nROUND $i\n"
  for j in {1..10}; do
    python ./run_lr.py -seed $i&
  done
  wait
done # 2>/dev/null