#!/bin/bash

mkdir -p train_dir_wikiqa

python runner.py --gpu_id 0 \
--train_dir train_dir_wikiqa \
--exp_mode train_debug --data_mode wikiqa \
--max_sent_length 100 --max_doc_length 30 \
--max_filter_length 8 --min_filter_length 5 \
--batch_size 44 --learning_rate 0.00017326216824334402 \
--size 100 --sentembed_size 166 \
--use_dropout False \
--max_gradient_norm -1 \
--train_epoch_crossentropy 40 \
--training_checkpoint 5 2>err2.log
