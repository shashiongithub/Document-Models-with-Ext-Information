####################################
# Author: Ronald Cardenas
# Date: July 2017
# Project: Document Modeling with External Attention for Sentence Extraction

####################################


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os, sys
import cPickle as pickle
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import SVC
from my_flags import FLAGS
from data_utils import DataProcessor
from my_model import MY_Model
from model_utils import convert_logits_to_softmax
from train_test_utils import batch_predict_with_a_model, batch_load_data
import argparse
import numpy as np
from sklearn.preprocessing import normalize
from utils_extra import *

import pdb



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Classification rutines for input: frozen SIDENET output + extra_features')
    parser.add_argument('-force_reading','--force_reading', help='Force reading dataset from preprocessed files',action="store_true", default=False)
    parser.add_argument('dataset',help='Dataset to use / mode of FLAGS setup',type=str,default="newsqa")
    parser.add_argument('predictions_dir',help='Directory where emb and probs are',type=str,default="../sidenet/train_dir_newsqa")

    args = parser.parse_args()
    mode = args.dataset

    pred_dir = os.path.abspath(args.predictions_dir)

    if mode=="newsqa":
        FLAGS.train_dir = args.predictions_dir
        FLAGS.load_prediction = 8
        FLAGS.max_sent_length = 50
        FLAGS.max_doc_length = 64
        C_param = 10
    elif mode=="squad":
        FLAGS.train_dir = args.predictions_dir
        FLAGS.load_prediction = 10
        FLAGS.max_sent_length = 80
        FLAGS.max_doc_length = 16
        C_param = 10.0
    elif mode=='wikiqa':
        FLAGS.train_dir = args.predictions_dir
        FLAGS.load_prediction = 10
        FLAGS.pretrained_wordembedding_orgdata = "../datasets/word_emb/1-billion-word-language-modeling-benchmark-r13output.word2vec.vec"
        FLAGS.preprocessed_data_directory = "../datasets/preprocessed_data"
        FLAGS.max_sent_length = 100
        FLAGS.max_doc_length = 30
        C_param = 100 
    elif mode=="msmarco":
        FLAGS.train_dir = args.predictions_dir
        FLAGS.load_prediction = 14
        FLAGS.max_sent_length = 150
        FLAGS.max_doc_length = 10
        C_param = 0.01

    FLAGS.force_reading = args.force_reading
    FLAGS.data_mode = mode
    FLAGS.use_ablated = True
    FLAGS.norm_extra_feats = True
    FLAGS.decorrelate_extra_feats = False

    suf = '_as' if mode=='squad' else ''
    wang_preds_folder = '../wang/data/'+mode+suf+'/'
    if mode=='msmarco':
        indexes = open("../wang/data/"+mode+"/train.filtered",'r').read().strip('\n').split('\n')
        ign_indexes_train = [int(x)-1 for x in indexes]
        indexes = open("../wang/data/"+mode+"/dev.filtered",'r').read().strip('\n').split('\n')
        ign_indexes_val = [int(x)-1 for x in indexes]
        indexes = open("../wang/data/"+mode+"/test.filtered",'r').read().strip('\n').split('\n')
        ign_indexes_test = [int(x)-1 for x in indexes]
    else:
        ign_indexes_train,ign_indexes_val,ign_indexes_test = [],[],[]
    
    print("Reading vocabulary...")
    vocab_dict, word_embedding_array = DataProcessor().prepare_vocab_embeddingdict()
    print("Reading training set...")
    train_data = DataProcessor().prepare_news_data(vocab_dict, data_type="training") # subsampled
    train_batch = batch_load_data(train_data)
    # data in whole batch with padded matrixes
    print("Reading validation set...")
    val_batch = batch_load_data(DataProcessor().prepare_news_data(vocab_dict,
                                                                  data_type="validation",
                                                                  normalizer=train_data.normalizer,
                                                                  pca_model=train_data.pca_model))
    
    print("Reading test set...")
    test_batch = batch_load_data(DataProcessor().prepare_news_data(vocab_dict,
                                                                  data_type="test",
                                                                  normalizer=train_data.normalizer,
                                                                  pca_model=train_data.pca_model))
    del train_data
    
    probs = np.load(os.path.join(pred_dir,"step-a.model.ckpt.epoch-%d.training-prediction.npy" % FLAGS.load_prediction))
    wang_pred_train = load_trec_prediction(os.path.join(wang_preds_folder,'training.trec_pred'))
    xtrain,ytrain,train_doc_lens = get_sentence_lvl_dataset(probs,train_batch,wang_pred_train,mode,ign_indexes_train)

    probs = np.load(os.path.join(pred_dir, "step-a.model.ckpt.epoch-%d.validation-prediction.npy" % FLAGS.load_prediction))
    wang_pred_val = load_trec_prediction(os.path.join(wang_preds_folder,'validation.trec_pred'))
    xval  ,yval,  val_doc_lens = get_sentence_lvl_dataset(probs,val_batch,wang_pred_val,mode,ign_indexes_val)

    probs = np.load(os.path.join(pred_dir, "step-a.model.ckpt.epoch-%d.test-prediction.npy" % FLAGS.load_prediction))
    wang_pred_test = load_trec_prediction(os.path.join(wang_preds_folder,'test.trec_pred'))
    xtest,ytest,test_doc_lens = get_sentence_lvl_dataset(probs,test_batch,wang_pred_test,mode,ign_indexes_test)

    print("Training* dataset size:",xtrain.shape)
    print("Validation* dataset size:",xval.shape)
    print("Test* dataset size:",xtest.shape)

    del train_batch
    del val_batch
    del probs
    del test_batch


    ##################################################################################################
    print("LR w C:%f " % C_param)    
    model = linear_model.LogisticRegression(C=C_param,n_jobs=30,random_state=42,penalty="l2")
    
    model.fit(xtrain,ytrain)
    pred_train = model.predict_proba(xtrain)[:,1]
    pred_val = model.predict_proba(xval)[:,1]
    
    
    tacc = accuracy(ytrain,pred_train,train_doc_lens)
    vacc = accuracy(yval,pred_val,val_doc_lens)

    tmrr = mrr_metric(ytrain,pred_train,train_doc_lens)
    vmrr = mrr_metric(yval,pred_val,val_doc_lens,"validation")

    tmap = map_score(ytrain,pred_train,train_doc_lens)
    vmap = map_score(yval,pred_val,val_doc_lens,"validation")

    
    pred_test = model.predict_proba(xtest)[:,1]
    test_acc = accuracy(ytest,pred_test,test_doc_lens)
    test_mrr = mrr_metric(ytest,pred_test,test_doc_lens,"test")
    test_map = map_score(ytest,pred_test,test_doc_lens,"test")

    weights = model.coef_
    bias = model.intercept_
    niters = model.n_iter_

    print("Sidenet")
    print("  Validation: acc: %.4f | mrr: %.4f | map: %.4f" % \
            (accuracy  (yval,xval[:,0],val_doc_lens),\
             mrr_metric(yval,xval[:,0],val_doc_lens,"validation"),\
             map_score (yval,xval[:,0],val_doc_lens,"validation")  ) )

    print("  Test      : acc: %.4f | mrr: %.4f | map: %.4f" % \
            (accuracy  (ytest,xtest[:,0],test_doc_lens),\
             mrr_metric(ytest,xtest[:,0],test_doc_lens,"test"),\
             map_score (ytest,xtest[:,0],test_doc_lens,"test")  ) )
    print()

    
    print("Sidenet + ISF Filtering on decoding:")
    
    print("  Training  : acc: %.4f | mrr: %.4f | map: %.4f" % \
            (accuracy  (ytrain,xtrain[:,0],train_doc_lens,xtrain[:,1]),\
             mrr_metric(ytrain,xtrain[:,0],train_doc_lens,xtrain[:,1]),\
             map_score (ytrain,xtrain[:,0],train_doc_lens,xtrain[:,1])  ) )
    
    print("  Validation: acc: %.4f | mrr: %.4f | map: %.4f" % \
            (accuracy  (yval,xval[:,0],val_doc_lens,xval[:,1]),\
             mrr_metric(yval,xval[:,0],val_doc_lens,xval[:,1]),\
             map_score (yval,xval[:,0],val_doc_lens,xval[:,1])  ) )
    print("  Test      : acc: %.4f | mrr: %.4f | map: %.4f" % \
            (accuracy  (ytest,xtest[:,0],test_doc_lens,xtest[:,1]),\
             mrr_metric(ytest,xtest[:,0],test_doc_lens,xtest[:,1]),\
             map_score (ytest,xtest[:,0],test_doc_lens,xtest[:,1])  ) )
    print()
    

    print("LR : Sidenet + CompAggr + Loc ISF")
    print("  Training  : acc: %.4f | mrr: %.4f | map: %.4f" % (tacc,tmrr,tmap) )
    print("  Validation: acc: %.4f | mrr: %.4f | map: %.4f" % (vacc,vmrr,vmap) )    
    print("  Test      : acc: %.4f | mrr: %.4f | map: %.4f" % (test_acc,test_mrr,test_map) )

    print("  Weights:",weights)
    print("  Bias:",bias)
    print("  N_iters:",niters)

    
