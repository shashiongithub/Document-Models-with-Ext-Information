#!/bin/bash

gpu=$1
train_dir=$2
model=$3

echo "Training set"
echo "--------------------------------------------------------------------------------------------------------------"
CUDA_VISIBLE_DEVICES=$gpu python runner.py \
--gpu_id $gpu --exp_mode test_train --model_to_load $model \
--train_dir $train_dir \
--data_mode newsqa \
--max_sent_length 50 --max_doc_length 64 \
--max_filter_length 8 --min_filter_length 5 \
--batch_size 64 --learning_rate 0.0001 \
--mlp_size 100 --sentembed_size 348 \
--max_gradient_norm 15 \
--use_dropout False \
--save_preds True \
--max_gradient_norm 15 2>err.log


echo ""
echo "Validation set"
echo "--------------------------------------------------------------------------------------------------------------"
CUDA_VISIBLE_DEVICES=$gpu python runner.py \
--gpu_id $gpu --exp_mode test_val --model_to_load $model \
--train_dir $train_dir \
--data_mode newsqa \
--max_sent_length 50 --max_doc_length 64 \
--max_filter_length 8 --min_filter_length 5 \
--batch_size 64 --learning_rate 0.0001 \
--mlp_size 100 --sentembed_size 348 \
--max_gradient_norm 15 \
--use_dropout False \
--save_preds True \
--max_gradient_norm 15 2>err.log



echo ""
echo "Test set"
echo "--------------------------------------------------------------------------------------------------------------"
CUDA_VISIBLE_DEVICES=$gpu python runner.py \
--gpu_id $gpu --exp_mode test --model_to_load $model \
--train_dir $train_dir \
--data_mode newsqa \
--max_sent_length 50 --max_doc_length 64 \
--max_filter_length 8 --min_filter_length 5 \
--batch_size 64 --learning_rate 0.0001 \
--mlp_size 100 --sentembed_size 348 \
--max_gradient_norm 15 \
--use_dropout False \
--save_preds True \
--max_gradient_norm 15 2>err.log
