#!/bin/bash

gpu=$1
train_dir=$2
model=$3

echo "Training set"
echo "XNET -------------------------------------"
CUDA_VISIBLE_DEVICES=$gpu python3 runner.py \
--gpu_id $gpu --exp_mode test_train --model_to_load $model \
--train_dir $train_dir \
--data_mode wikiqa \
--max_sent_length 100 --max_doc_length 30 \
--batch_size 22 --learning_rate 0.000246818447 \
--size 447 --sentembed_size 80 \
--max_filter_length 8 --min_filter_length 5 \
--use_dropout False \
--save_preds True \
--max_gradient_norm 10 2>err.log


echo ""
echo "Validation set"
echo "XNET -------------------------------------"

CUDA_VISIBLE_DEVICES=$gpu python3 runner.py \
--gpu_id $gpu --exp_mode test_val --model_to_load $model \
--train_dir $train_dir \
--data_mode wikiqa \
--max_sent_length 100 --max_doc_length 30 \
--batch_size 22 --learning_rate 0.000246818447 \
--size 447 --sentembed_size 80 \
--max_filter_length 8 --min_filter_length 5 \
--use_dropout False \
--save_preds True \
--max_gradient_norm 10 2>err.log


echo ""
echo "XNET top5 -------------------------------------"
# topK filtered
CUDA_VISIBLE_DEVICES=$gpu python3 runner.py \
--gpu_id $gpu --exp_mode test_val --model_to_load $model \
--train_dir $train_dir \
--data_mode wikiqa \
--max_sent_length 100 --max_doc_length 30 \
--batch_size 22 --learning_rate 0.000246818447 \
--size 447 --sentembed_size 80 \
--max_filter_length 8 --min_filter_length 5 \
--use_dropout False \
--max_gradient_norm 10 \
--filtered_setting True --topK 5 2>err.log

echo ""
echo "Test set"
echo "XNET -------------------------------------"
CUDA_VISIBLE_DEVICES=$gpu python3 runner.py \
--gpu_id $gpu --exp_mode test --model_to_load $model \
--train_dir $train_dir \
--data_mode wikiqa \
--max_sent_length 100 --max_doc_length 30 \
--batch_size 22 --learning_rate 0.000246818447 \
--size 447 --sentembed_size 80 \
--max_filter_length 8 --min_filter_length 5 \
--use_dropout False \
--save_preds True \
--max_gradient_norm 10 2>err.log


echo ""
echo "XNET top5 -------------------------------------"
CUDA_VISIBLE_DEVICES=$gpu python3 runner.py \
--gpu_id $gpu --exp_mode test --model_to_load $model \
--train_dir $train_dir \
--data_mode wikiqa \
--max_sent_length 100 --max_doc_length 30 \
--batch_size 22 --learning_rate 0.000246818447 \
--size 447 --sentembed_size 80 \
--max_filter_length 8 --min_filter_length 5 \
--use_dropout False \
--max_gradient_norm 10 \
--filtered_setting True --topK 5 2>err.log
