# Document Modeling with External Information for Sentence Extraction

This repository contains the code neccesary to reproduce the results in the paper: 

Document Modeling with External Attention for Sentence Extraction, Shashi Narayan, Ronald Cardenas, Nikos Papasarantopoulos, Shay B. Cohen, Mirella Lapata, Jiangsheng Yu and Yi Chang, ACL 2018, Melbourne, Australia.



## Extractive Summarization

To train XNet+ (Title + Caption), run:
> python document_summarizer_gpu2.py --max_title_length 1 --max_image_length 10 --train_dir <my-training-dir> --model_to_load 8 --exp_mode train

from extractive_summ/.


## Answer Selection


1. Datasets and Resources

a) NewsQA

Download the combined dataset from: https://datasets.maluuba.com/NewsQA/dl

Download splitting scripts from NewsQA repo: https://github.com/Maluuba/newsqa

b) SQuAD: https://rajpurkar.github.io/SQuAD-explorer/

c) WikiQA: https://www.microsoft.com/en-us/download/details.aspx?id=52419

d) MarcoMS: http://www.msmarco.org/dataset.aspx

e) 1 billion words benchmark: http://www.statmt.org/lm-benchmark/


2. Preprocessing

First, train word embeddings on the 1BW benchmark using word2vec and place the files on answer_selection/datasets/word_emb.

Generate the score files (IDF, ISF, word counts) for each dataset by running 

> python reformat_corpus.py

from answer_selection/datasets/<dataset>/

The preprocessed files will be placed in the folder: answer_selection/datasets/preprocessed_data/<dataset>


3. Training

Run the scripts run_<dataset> in each model folder for training.


4. Evaluation

Run the scripts eval_<dataset> in each model folder for training.

