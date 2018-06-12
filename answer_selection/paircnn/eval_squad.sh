#!/bin/bash

gpu=$1
train_dir=$2
model=$3

echo "Training set"
echo "--------------------------------------------------------------------------------------------------------------"
CUDA_VISIBLE_DEVICES=$gpu python document_summarizer_gpu.py \
--gpu_id $gpu --exp_mode test_train --model_to_load $model \
--train_dir $train_dir \
--data_mode squad \
--max_sent_length 80 --max_doc_length 16 \
--max_filter_length 8 --min_filter_length 5 \
--batch_size 20 --learning_rate 0.0017494383670581671 \
--size 100 --sentembed_size 799 \
--max_gradient_norm -1 \
--use_dropout False \
--save_preds True  2>err.log


echo ""
echo "Validation set"
echo "--------------------------------------------------------------------------------------------------------------"
CUDA_VISIBLE_DEVICES=$gpu python document_summarizer_gpu.py \
--gpu_id $gpu --exp_mode test_val --model_to_load $model \
--train_dir $train_dir \
--data_mode squad \
--max_sent_length 80 --max_doc_length 16 \
--max_filter_length 8 --min_filter_length 5 \
--batch_size 20 --learning_rate 0.0017494383670581671 \
--size 100 --sentembed_size 799 \
--max_gradient_norm -1 \
--use_dropout False \
--save_preds True  2>err.log



echo ""
echo "Test set"
echo "--------------------------------------------------------------------------------------------------------------"
CUDA_VISIBLE_DEVICES=$gpu python document_summarizer_gpu.py \
--gpu_id $gpu --exp_mode test --model_to_load $model \
--train_dir $train_dir \
--data_mode squad \
--max_sent_length 80 --max_doc_length 16 \
--max_filter_length 8 --min_filter_length 5 \
--batch_size 20 --learning_rate 0.0017494383670581671 \
--size 100 --sentembed_size 799 \
--max_gradient_norm -1 \
--use_dropout False \
--save_preds True  2>err.log