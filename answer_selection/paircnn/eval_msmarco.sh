#!/bin/bash

gpu=$1
train_dir=$2
model=$3

echo "Training set"
CUDA_VISIBLE_DEVICES=$gpu python runner.py \
--gpu_id $gpu --exp_mode test_train --model_to_load $model \
--train_dir $train_dir \
--data_mode msmarco \
--max_sent_length 150 --max_doc_length 10 \
--max_filter_length 8 --min_filter_length 5 \
--batch_size 50 --learning_rate 1.9126116867403927e-05 \
--size 121 --sentembed_size 531 \
--max_gradient_norm -1 \
--use_dropout False \
--save_preds True  2>err.log


echo ""
echo "Validation set"
echo "Sidenet"
echo "--------------------------------------------------------------------------------------------------------------"
CUDA_VISIBLE_DEVICES=$gpu python runner.py \
--gpu_id $gpu --exp_mode test_val --model_to_load $model \
--train_dir $train_dir \
--data_mode msmarco \
--max_sent_length 150 --max_doc_length 10 \
--max_filter_length 8 --min_filter_length 5 \
--batch_size 50 --learning_rate 1.9126116867403927e-05 \
--size 121 --sentembed_size 531 \
--max_gradient_norm -1 \
--use_dropout False \
--save_preds True  2>err.log



echo ""
echo "Test set"
echo "--------------------------------------------------------------------------------------------------------------"
CUDA_VISIBLE_DEVICES=$gpu python runner.py \
--gpu_id $gpu --exp_mode test --model_to_load $model \
--train_dir $train_dir \
--data_mode msmarco \
--max_sent_length 150 --max_doc_length 10 \
--max_filter_length 8 --min_filter_length 5 \
--batch_size 50 --learning_rate 1.9126116867403927e-05 \
--size 121 --sentembed_size 531 \
--max_gradient_norm -1 \
--use_dropout False \
--save_preds True  2>err.log