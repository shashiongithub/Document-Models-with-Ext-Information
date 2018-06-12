#!/bin/bash

mkdir -p train_dir_msmarco

python runner.py --gpu_id 0 \
--train_dir train_dir_msmarco \
--exp_mode train \
--data_mode msmarco \
--max_sent_length 150 --max_doc_length 10 \
--max_filter_length 8 --min_filter_length 5 \
--batch_size 62 --learning_rate 2.1082974294910347e-05 \
--size 1521 --sentembed_size 854 \
--use_dropout False \
--max_gradient_norm 10 \
--training_checkpoint 5 \
--train_epoch_crossentropy 15 2>err.log

