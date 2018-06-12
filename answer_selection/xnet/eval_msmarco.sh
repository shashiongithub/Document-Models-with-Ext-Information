#!/bin/bash

gpu=$1
train_dir=$2
model=$3

echo "Training set"
echo "XNET -------------------------------------"

CUDA_VISIBLE_DEVICES=$gpu python document_summarizer_gpu.py \
--gpu_id $gpu --exp_mode test_train --model_to_load $model \
--train_dir $train_dir \
--data_mode msmarco \
--max_sent_length 150 --max_doc_length 10 \
--max_filter_length 8 --min_filter_length 5 \
--batch_size 62 --learning_rate 2.1082974294910347e-05 \
--size 1521 --sentembed_size 854 \
--use_dropout False \
--save_preds True \
--max_gradient_norm 10 2>err.log


echo ""
echo "Validation set"
echo "XNET -------------------------------------"

CUDA_VISIBLE_DEVICES=$gpu python document_summarizer_gpu.py \
--gpu_id $gpu --exp_mode test_val --model_to_load $model \
--train_dir $train_dir \
--data_mode msmarco \
--max_sent_length 150 --max_doc_length 10 \
--max_filter_length 8 --min_filter_length 5 \
--batch_size 62 --learning_rate 2.1082974294910347e-05 \
--size 1521 --sentembed_size 854 \
--use_dropout False \
--save_preds True \
--max_gradient_norm 10 2>err.log


echo ""
echo "XNET top5 -------------------------------------"

CUDA_VISIBLE_DEVICES=$gpu python document_summarizer_gpu.py \
--gpu_id $gpu --exp_mode test_val --model_to_load $model \
--train_dir $train_dir \
--filtered_setting True --topK 5 \
--data_mode msmarco \
--max_sent_length 150 --max_doc_length 10 \
--max_filter_length 8 --min_filter_length 5 \
--batch_size 62 --learning_rate 2.1082974294910347e-05 \
--size 1521 --sentembed_size 854 \
--use_dropout False \
--max_gradient_norm 10 2>err.log




echo ""
echo "Test set"
echo "XNET -------------------------------------"

CUDA_VISIBLE_DEVICES=$gpu python document_summarizer_gpu.py \
--gpu_id $gpu --exp_mode test --model_to_load $model \
--train_dir $train_dir \
--data_mode msmarco \
--max_sent_length 150 --max_doc_length 10 \
--max_filter_length 8 --min_filter_length 5 \
--batch_size 62 --learning_rate 2.1082974294910347e-05 \
--size 1521 --sentembed_size 854 \
--use_dropout False \
--save_preds True \
--max_gradient_norm 10 2>err.log




echo ""
echo "XNET top5 -------------------------------------"

CUDA_VISIBLE_DEVICES=$gpu python document_summarizer_gpu.py \
--gpu_id $gpu --exp_mode test --model_to_load $model \
--train_dir $train_dir \
--filtered_setting True --topK 5 \
--data_mode msmarco \
--max_sent_length 150 --max_doc_length 10 \
--max_filter_length 8 --min_filter_length 5 \
--batch_size 62 --learning_rate 2.1082974294910347e-05 \
--size 1521 --sentembed_size 854 \
--use_dropout False \
--max_gradient_norm 10 2>err.log

