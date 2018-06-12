#!/bin/bash

mkdir -p train_dir_squad

python runner.py --gpu_id 0 \
--train_dir train_dir_squad \
--exp_mode train \
--data_mode squad \
--max_sent_length 80 --max_doc_length 16 \
--max_filter_length 8 --min_filter_length 5 \
--batch_size 20 --learning_rate 0.0017494383670581671 \
--size 100 --sentembed_size 799 \
--use_dropout False \
--max_gradient_norm -1 \
--train_epoch_crossentropy 20 \
--training_checkpoint 5