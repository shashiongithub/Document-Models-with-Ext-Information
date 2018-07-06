# Document Modeling with External Information for Sentence Extraction:
## Answer Selection


1. Download the datasets
a) NewsQA
Download the combined dataset from: https://datasets.maluuba.com/NewsQA/dl
Download splitting scripts from NewsQA repo: https://github.com/Maluuba/newsqa

b) SQuAD: https://rajpurkar.github.io/SQuAD-explorer/

c) WikiQA: https://www.microsoft.com/en-us/download/details.aspx?id=52419

d) MarcoMS: http://www.msmarco.org/dataset.aspx

e) 1 billion words benchmark: http://www.statmt.org/lm-benchmark/

2. Datasets preprocessing

First, train word embeddings on the 1BW benchmark using word2vec and place the files on datasets/word_emb.

Generate the score files (IDF, ISF, word counts) for each dataset by running 

> python datasets/<dataset>/reformat_corpus.py

The preprocessed files will be placed in the folder: datasets/preprocessed_data/<dataset>


2. Training
Run the scripts run_<dataset> in each model folder for training.


3. Evaluation
Run the scripts eval_<dataset> in each model folder for training.

