#!/bin/bash
./train.sh 4 ./checkpoints_depth \
              --ndepths "48,32,8" \
              --depth_inter_r "4,2,1" \
              --dlossw "0.5,1.0,2.0" \
              --batch_size 2 \
              --eval_freq 3 \
              --epochs 64 \
              --use_lq_depth
