#!/bin/bash

mkdir -p train_dir_newsqa

python runnner.py --gpu_id 0 \
--train_dir train_dir_newsqa \
--exp_mode train \
--max_sent_length 50 --max_doc_length 64 \
--max_filter_length 8 --min_filter_length 5 \
--batch_size 64 --learning_rate 6.351693395598085e-05 \
--size 1021 --sentembed_size 558 \
--use_dropout False \
--max_gradient_norm 10 \
--training_checkpoint 5 \
--train_epoch_crossentropy 15

