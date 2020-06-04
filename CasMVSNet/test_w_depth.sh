#!/bin/bash
export PATH=/opt/conda/bin:$PATH

TESTPATH="/data/dtu_test"
TESTLIST="lists/dtu/test.txt"
CKPT_FILE=$1
python3 test.py --dataset=general_eval \
                --batch_size=1 \
                --testpath=$TESTPATH \
                --testlist=$TESTLIST \
                --loadckpt $CKPT_FILE \
                --num_consistent=1 \
                --prob_threshold=0.5 ${@:2} \
                --use_lq_depth
