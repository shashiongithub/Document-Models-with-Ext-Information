#!/bin/bash

mkdir -p train_dir_msmarco


python runner.py --gpu_id 0 \
--train_dir train_dir_msmarco \
--exp_mode train_debug \
--data_mode msmarco \
--max_sent_length 150 --max_doc_length 10 \
--max_filter_length 8 --min_filter_length 5 \
--batch_size 50 --learning_rate 1.9126116867403927e-05 \
--size 121 --sentembed_size 531 \
--preprocessed_data_directory ../datasets/preprocessed_data \
--use_dropout False \
--max_gradient_norm -1 \
--training_checkpoint 5 2>err.log

# sentemb -> 528