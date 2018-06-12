#!/bin/bash

gpu=$1
train_dir=$2
model=$3

echo "Training set"
echo "XNET -------------------------------------"
CUDA_VISIBLE_DEVICES=$gpu python runner.py \
--gpu_id $gpu --exp_mode test_train --model_to_load $model \
--train_dir $train_dir \
--data_mode wikiqa \
--max_sent_length 100 --max_doc_length 30 \
--max_filter_length 8 --min_filter_length 5 \
--batch_size 44 --learning_rate 0.00017326216824334402 \
--size 100 --sentembed_size 164 \
--use_dropout False \
--save_preds True \
--max_gradient_norm -1 2>err.log


echo ""
echo "Validation set"
echo "XNET -------------------------------------"
CUDA_VISIBLE_DEVICES=$gpu python runner.py \
--gpu_id $gpu --exp_mode test_val --model_to_load $model \
--train_dir $train_dir \
--data_mode wikiqa \
--max_sent_length 100 --max_doc_length 30 \
--max_filter_length 8 --min_filter_length 5 \
--batch_size 44 --learning_rate 0.00017326216824334402 \
--size 100 --sentembed_size 164 \
--use_dropout False \
--save_preds True \
--max_gradient_norm -1 2>err.log


echo ""
echo "XNET top5 -------------------------------------"

CUDA_VISIBLE_DEVICES=$gpu python runner.py \
--gpu_id $gpu --exp_mode test_val --model_to_load $model \
--train_dir $train_dir \
--data_mode wikiqa \
--filtered_setting True --topK 5 \
--max_sent_length 100 --max_doc_length 30 \
--max_filter_length 8 --min_filter_length 5 \
--batch_size 44 --learning_rate 0.00017326216824334402 \
--size 100 --sentembed_size 164 \
--use_dropout False \
--max_gradient_norm -1 2>err.log

echo ""
echo "Test set"
echo "XNET -------------------------------------"

CUDA_VISIBLE_DEVICES=$gpu python runner.py \
--gpu_id $gpu --exp_mode test --model_to_load $model \
--train_dir $train_dir \
--data_mode wikiqa \
--max_sent_length 100 --max_doc_length 30 \
--max_filter_length 8 --min_filter_length 5 \
--batch_size 44 --learning_rate 0.00017326216824334402 \
--size 100 --sentembed_size 164 \
--use_dropout False \
--save_preds True \
--max_gradient_norm -1 2>err.log


echo ""
echo "XNET top5 -------------------------------------"

CUDA_VISIBLE_DEVICES=$gpu python runner.py \
--gpu_id $gpu --exp_mode test --model_to_load $model \
--train_dir $train_dir \
--data_mode wikiqa \
--filtered_setting True --topK 5 \
--max_sent_length 100 --max_doc_length 30 \
--max_filter_length 8 --min_filter_length 5 \
--batch_size 44 --learning_rate 0.00017326216824334402 \
--size 100 --sentembed_size 164 \
--use_dropout False \
--max_gradient_norm -1 2>err.log
