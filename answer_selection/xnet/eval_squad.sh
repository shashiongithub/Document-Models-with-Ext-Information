#!/bin/bash

gpu=$1
train_dir=$2
model=$3

echo "Training set"
echo "XNET -------------------------------------"

CUDA_VISIBLE_DEVICES=$gpu python document_summarizer_gpu.py \
--gpu_id $gpu --exp_mode test_train --model_to_load $model \
--train_dir $train_dir \
--data_mode squad \
--max_sent_length 80 --max_doc_length 16 \
--max_filter_length 8 --min_filter_length 5 \
--batch_size 56 --learning_rate 5.3578003238215205e-05 \
--size 1446 --sentembed_size 371 \
--use_dropout False \
--save_preds True \
--max_gradient_norm 10 2>err.log


echo ""
echo "Validation set"
echo "XNET -------------------------------------"

CUDA_VISIBLE_DEVICES=$gpu python document_summarizer_gpu.py \
--gpu_id $gpu --exp_mode test_val --model_to_load $model \
--train_dir $train_dir \
--data_mode squad \
--max_sent_length 80 --max_doc_length 16 \
--max_filter_length 8 --min_filter_length 5 \
--batch_size 56 --learning_rate 5.3578003238215205e-05 \
--size 1446 --sentembed_size 371 \
--use_dropout False \
--save_preds True \
--max_gradient_norm 10 2>err.log


echo ""
echo "XNET top5 -------------------------------------"

CUDA_VISIBLE_DEVICES=$gpu python document_summarizer_gpu.py \
--gpu_id $gpu --exp_mode test_val --model_to_load $model \
--train_dir $train_dir \
--data_mode squad \
--max_sent_length 80 --max_doc_length 16 \
--max_filter_length 8 --min_filter_length 5 \
--batch_size 56 --learning_rate 5.3578003238215205e-05 \
--size 1446 --sentembed_size 371 \
--use_dropout False \
--save_preds True \
--max_gradient_norm 10 \
--filtered_setting True --topK 5 2>err.log




echo ""
echo "Test set"
echo "XNET -------------------------------------"

CUDA_VISIBLE_DEVICES=$gpu python document_summarizer_gpu.py \
--gpu_id $gpu --exp_mode test --model_to_load $model \
--train_dir $train_dir \
--data_mode squad \
--max_sent_length 80 --max_doc_length 16 \
--max_filter_length 8 --min_filter_length 5 \
--batch_size 56 --learning_rate 5.3578003238215205e-05 \
--size 1446 --sentembed_size 371 \
--use_dropout False \
--save_preds True \
--max_gradient_norm 10 2>err.log


echo ""
echo "XNET top5 -------------------------------------"

CUDA_VISIBLE_DEVICES=$gpu python document_summarizer_gpu.py \
--gpu_id $gpu --exp_mode test --model_to_load $model \
--train_dir $train_dir \
--data_mode squad \
--max_sent_length 80 --max_doc_length 16 \
--max_filter_length 8 --min_filter_length 5 \
--batch_size 56 --learning_rate 5.3578003238215205e-05 \
--size 1446 --sentembed_size 371 \
--use_dropout False \
--save_preds True \
--max_gradient_norm 10 \
--filtered_setting True --topK 5 2>err.log



