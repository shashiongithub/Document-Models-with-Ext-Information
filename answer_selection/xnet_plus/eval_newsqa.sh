#!/bin/bash

gpu=$1
train_dir=$2
model=$3

echo "Training set"
echo "XNET -------------------------------------"

Sidenet orig
CUDA_VISIBLE_DEVICES=$gpu python runner.py \
--gpu_id $gpu --exp_mode test_train --model_to_load $model \
--train_dir $train_dir \
--data_mode newsqa \
--max_sent_length 50 --max_doc_length 64 \
--max_filter_length 8 --min_filter_length 5 \
--batch_size 42 --learning_rate 1.3798932121802611e-05 \
--size 100 --sentembed_size 1024 \
--use_dropout False \
--save_preds True \
--max_gradient_norm -1 2>err.log


echo ""
echo "Validation set"
echo "XNET -------------------------------------"

CUDA_VISIBLE_DEVICES=$gpu python runner.py \
--gpu_id $gpu --exp_mode test_val --model_to_load $model \
--train_dir $train_dir \
--data_mode newsqa \
--max_sent_length 50 --max_doc_length 64 \
--max_filter_length 8 --min_filter_length 5 \
--batch_size 42 --learning_rate 1.3798932121802611e-05 \
--size 100 --sentembed_size 1024 \
--use_dropout False \
--save_preds True \
--max_gradient_norm -1 2>err.log


echo ""
echo "XNET top5 -------------------------------------"

# Sidenet orig topK filtered
CUDA_VISIBLE_DEVICES=$gpu python runner.py \
--gpu_id $gpu --exp_mode test_val --model_to_load $model \
--train_dir $train_dir \
--data_mode newsqa \
--filtered_setting True --topK 5 \
--max_sent_length 50 --max_doc_length 64 \
--max_filter_length 8 --min_filter_length 5 \
--batch_size 42 --learning_rate 1.3798932121802611e-05 \
--size 100 --sentembed_size 1024 \
--use_dropout False \
--max_gradient_norm -1 2>err.log

echo ""
echo "Test set"
echo "XNET -------------------------------------"

CUDA_VISIBLE_DEVICES=$gpu python runner.py \
--gpu_id $gpu --exp_mode test --model_to_load $model \
--train_dir $train_dir \
--data_mode newsqa \
--max_sent_length 50 --max_doc_length 64 \
--max_filter_length 8 --min_filter_length 5 \
--batch_size 42 --learning_rate 1.3798932121802611e-05 \
--size 100 --sentembed_size 1024 \
--use_dropout False \
--save_preds True \
--max_gradient_norm -1 2>err.log
