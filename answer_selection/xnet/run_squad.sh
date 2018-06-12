#!/bin/bash

mkdir -p train_dir_squad

python runner.py --gpu_id 0 \
--train_dir train_dir_squad \
--exp_mode train_debug --data_mode squad \
--max_sent_length 80 --max_doc_length 16 \
--max_filter_length 8 --min_filter_length 5 \
--batch_size 56 --learning_rate 5.3578003238215205e-05 \
--size 1446 --sentembed_size 371 \
--preprocessed_data_directory ../datasets/preprocessed_data \
--use_dropout False \
--max_gradient_norm 10 \
--training_checkpoint 5 2>err.log

