####################################
# Author: Ronald Cardenas
# Date: July 2017
# Project: Document Modeling with External Attention for Sentence Extraction

####################################


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('../../common')

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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import pdb

from utils_extra import *

C_param = 0.0001


def get_sentence_lvl_dataset_abl(probs,batch,cos_sim,wang_preds,mode,ign_indexes):
    prob_one = probs[:,:,0]
    labels = batch.labels[:,:,0]

    bs,ld = prob_one.shape
    cnt_score = batch.cnt_score
    isf_score = batch.isf_score
    idf_score = batch.idf_score
    locisf_score = batch.locisf_score
    sent_lens = batch.sent_lens # [bs,ld]

    doc_lens = batch.weights.sum(axis=1).astype(np.int32)
    X = []
    Y = []
    
    indexes = [x for x in range(bs) if x not in ign_indexes]
    if mode=='newsqa':
        indexes=indexes[:-1]
    filt_doc_len = []
    
    for idx,i in enumerate(indexes):
        doc_len = doc_lens[i]
        filt_doc_len.append(doc_len)
        if doc_len > len(wang_preds[idx]):
            pdb.set_trace()
        
        meta_x = np.vstack([
            prob_one[i,:doc_len],
            cos_sim[i,:doc_len],
            sent_lens[i,:doc_len],
            cnt_score[i,:doc_len],
            idf_score[i,:doc_len],
            isf_score[i,:doc_len],
            locisf_score[i,:doc_len],
            wang_preds[idx][:doc_len],
            ])
        
        X.append(meta_x)
        Y.extend(labels[i,:doc_len])

    X = np.hstack(X).T
    Y = np.array(Y,dtype=np.float32)

    return X,Y,filt_doc_len



def evaluate_model(xtrain,ytrain,train_doc_lens, xval,yval,val_doc_lens, args):
    if args.logreg:
        model = linear_model.LogisticRegression(C=C_param,n_jobs=30,random_state=42,penalty="l2")
    if args.svc:
        model = SVC(C=0.01,kernel="linear",random_state=42)
    model.fit(xtrain,ytrain)
    pred_train = model.predict_proba(xtrain)
    pred_val = model.predict_proba(xval)
    pred_train = pred_train[:,1]
    pred_val   = pred_val[:,1]
    tacc = accuracy(ytrain,pred_train,train_doc_lens)
    #tacc = map_score(ytrain,pred_train,train_doc_lens,'training')
    vacc = accuracy(yval,pred_val,val_doc_lens)
    #vacc = map_score(yval,pred_val,val_doc_lens,'validation')

    weights = model.coef_
    bias = model.intercept_
    niters = model.n_iter_

    return weights,bias,niters,tacc,vacc



def run_classifier(xtrain,xval,ytrain,yval,parameters,train_DL, val_DL, use_l1=False):
    vacc = -1
    vmap = -1
    best_C = -1
    norm = 'l1' if use_l1 else 'l2'
    
    for c in parameters["C"]:
        model = linear_model.LogisticRegression(C=c,n_jobs=30,random_state=42,penalty=norm)
        model.fit(xtrain,ytrain)
        pred_train = model.predict_proba(xtrain)
        pred_val = model.predict_proba(xval)
        pred_val   = pred_val[:,1]
        vacc_curr = accuracy(yval,pred_val,val_DL)
        if vacc_curr > vacc:
            vacc = vacc_curr
            best_C = c
    #END-FOR-C

    return vacc,best_C
    #return vmap,best_C



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Classification rutines for input: frozen SIDENET output + extra_features')
    parser.add_argument('-lr','--logreg', help='Use Logisting Regressor',action="store_true")
    parser.add_argument('-svc' ,'--svc', help='Use Support Vector Classifier',action="store_true")
    parser.add_argument('-rf','--rf', help='Use Random Forest',action="store_true")
    parser.add_argument('-force_reading','--force_reading', help='Force reading dataset from preprocessed files',action="store_true", default=False)
    parser.add_argument('dataset',help='Dataset to use / mode of FLAGS setup',type=str,default="newsqa")
    parser.add_argument('predictions_dir',help='Directory where emb and probs are',type=str,default="../sidenet/train_dir_newsqa")

    args = parser.parse_args()
    mode = args.dataset

    pred_dir = os.path.abspath(args.predictions_dir)

    if  not args.logreg and \
        not args.svc and \
        not args.rf:
        print("Please specify which model to use")
        sys.exit(0)

    if mode=="newsqa":
        FLAGS.train_dir = args.predictions_dir
        FLAGS.load_prediction = 8
        FLAGS.max_sent_length = 50
        FLAGS.max_doc_length = 64
    elif mode=="squad":
        FLAGS.train_dir = args.predictions_dir
        FLAGS.load_prediction = 10
        FLAGS.max_sent_length = 80
        FLAGS.max_doc_length = 16
    elif mode=="wikiqa":
        FLAGS.train_dir = args.predictions_dir
        FLAGS.load_prediction = 10
        FLAGS.max_sent_length = 100
        FLAGS.max_doc_length = 30
    elif mode=="msmarco":
        FLAGS.train_dir = args.predictions_dir
        FLAGS.load_prediction = 14
        FLAGS.max_sent_length = 150
        FLAGS.max_doc_length = 10

    FLAGS.pretrained_wordembedding_orgdata = "../datasets/word_emb/1-billion-word-language-modeling-benchmark-r13output.word2vec.vec"
    FLAGS.preprocessed_data_directory = "../datasets/preprocessed_data"

    FLAGS.force_reading = args.force_reading
    FLAGS.data_mode = mode
    FLAGS.use_ablated = True
    FLAGS.norm_extra_feats = True
    FLAGS.decorrelate_extra_feats = False

    suf = '_as' if mode=='squad' else ''
    wang_preds_folder = '../wang/data/'+mode+suf+'/'

    if mode=='msmarco':
        indexes = open("../wang/data/"+mode+"/train.filtered",'r').read().strip('\n').split('\n')
        ign_indexes_train = [int(x)-1 for x in indexes if x!='']
        indexes = open("../wang/data/"+mode+"/dev.filtered",'r').read().strip('\n').split('\n')
        ign_indexes_val = [int(x)-1 for x in indexes if x!='']
        indexes = open("../wang/data/"+mode+"/test.filtered",'r').read().strip('\n').split('\n')
        ign_indexes_test = [int(x)-1 for x in indexes if x!='']
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
    del train_data
    
    probs = np.load(os.path.join(pred_dir,"step-a.model.ckpt.epoch-%d.training-prediction.npy" % FLAGS.load_prediction))
    cos_sim = np.load(os.path.join(pred_dir,"step-a.model.ckpt.epoch-%d.training-cos_sim.npy" % FLAGS.load_prediction))
    wang_pred_train = load_trec_prediction(os.path.join(wang_preds_folder,'training.trec_pred'))

    xtrain,ytrain,train_doc_lens = get_sentence_lvl_dataset_abl(probs,train_batch,cos_sim,wang_pred_train,mode,ign_indexes_train)


    probs = np.load(os.path.join(pred_dir, "step-a.model.ckpt.epoch-%d.validation-prediction.npy" % FLAGS.load_prediction))
    cos_sim = np.load(os.path.join(pred_dir,"step-a.model.ckpt.epoch-%d.validation-cos_sim.npy" % FLAGS.load_prediction))
    wang_pred_val = load_trec_prediction(os.path.join(wang_preds_folder,'validation.trec_pred'))
    xval  ,yval,  val_doc_lens = get_sentence_lvl_dataset_abl(probs,val_batch,cos_sim,wang_pred_val,mode,ign_indexes_val)


    print("Training* dataset size:",xtrain.shape)
    print("Validation* dataset size:",xval.shape)

    del train_batch
    del val_batch
    del probs
    del cos_sim

    ##################################################################################################
    print("Complete model...")
    
    weights,bias,niters,tacc,vacc = evaluate_model(
        xtrain,ytrain,train_doc_lens,
        xval,yval,val_doc_lens,
        args
        )

    metric = "ACC"
    print("  Training %s: " % metric, tacc)
    print("  Validation %s:" % metric,vacc)
    print("  Weights:",weights)
    print("  Bias:",bias)
    print("  N_iters:",niters)

    ##################################################################################################
    feature_ablated = np.array([
        "sidenet_prob      ",
        "cosine similarity ",
        "sentence length   ",
        "word count        ",
        "IDF score         ",
        "ISF score         ",
        "Local ISF score   ",
        "CompAggr          ",
    ])
    chopped_x_train = xtrain
    chopped_x_val = xval
    n_feats = xtrain.shape[1]
    mask_feats = np.ones(n_feats)
    feat_indexes = list(range(n_feats))

    parameters = {'C':[1e-4,0.01,0.1,1,10,100,1e3,1e4,1e5]}
    
    for feats_left in range(n_feats,1,-1):
        print("----------------------------------------------------")
        accuracies = np.zeros(n_feats)
        for pos in range(feats_left):
            temp_indexes = feat_indexes[:pos] + feat_indexes[pos+1:]
            chopped_x_train = xtrain[:,temp_indexes]
            chopped_x_val   = xval[:,temp_indexes]
            vacc,best_c = run_classifier(chopped_x_train,chopped_x_val,ytrain,yval,parameters,train_doc_lens,val_doc_lens)
            accuracies[feat_indexes[pos]] = vacc
            print("Ablating feat.:%s | %s: %.4f , best_C: %f" % (feature_ablated[feat_indexes[pos]],metric,vacc,best_c) )
        to_ablate = np.argmax(accuracies[feat_indexes])
        # always include sidenet lol
        if feat_indexes[to_ablate]==0:
            accuracies[feat_indexes[to_ablate]] = 0
            to_ablate = np.argmax(accuracies[feat_indexes])    
        feat_indexes = feat_indexes[:to_ablate] + feat_indexes[to_ablate+1:]

        
