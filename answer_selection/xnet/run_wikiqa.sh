#!/bin/bash

mkdir -p train_dir_wikiqa

## no dropout, sigopt / wang sets
python runner.py --gpu_id 0 \
--train_dir train_dir_wikiqa \
--exp_mode train --data_mode wikiqa \
--max_sent_length 100 --max_doc_length 30 \
--batch_size 22 --learning_rate 0.000246818447 \
--size 447 --sentembed_size 80 \
--max_filter_length 8 --min_filter_length 5 \
--use_dropout False \
--max_gradient_norm 10 \
--train_epoch_crossentropy 30 \
--training_checkpoint 5
