#!/bin/bash

mkdir -p train_dir_newsqa

python runner.py --gpu_id 0 \
--train_dir train_dir_newsqa \
--exp_mode train_debug --data_mode newsqa \
--max_sent_length 50 --max_doc_length 64 \
--max_filter_length 8 --min_filter_length 5 \
--batch_size 42 --learning_rate 1.3798932121802611e-05 \
--size 100 --sentembed_size 1024 \
--use_dropout False \
--max_gradient_norm -1 \
--training_checkpoint 5 2>err.log
