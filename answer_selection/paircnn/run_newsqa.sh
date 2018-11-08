#!/bin/bash

mkdir -p train_dir_newsqa

CUDA_VISIBLE_DEVICES=0 python runner.py --gpu_id 0 \
--train_dir train_dir_newsqa --data_mode newsqa \
--exp_mode train \
--tie_break first \
--max_sent_length 50 --max_doc_length 64 \
--max_filter_length 8 --min_filter_length 5 \
--batch_size 64 --learning_rate 0.0001 \
--mlp_size 100 --sentembed_size 348 \
--max_gradient_norm 15 \
--training_checkpoint 20 \
--training_checkpoint 5

