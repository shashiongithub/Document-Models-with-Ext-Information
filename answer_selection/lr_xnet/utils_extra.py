####################################
# Author: Ronald Cardenas
# Date: July 2017
# Project: Document Modeling with External Attention for Sentence Extraction

####################################
import sys
sys.path.append('../../common')

from sklearn.metrics import average_precision_score as aps
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pdb
import subprocess as sp
import os
from my_flags import FLAGS

topK = 5
window = 0


def dump_trec_format(labels,scores,doc_lens):

    output = open("temp.trec_res",'w')
    qid = 1
    idx = 0
    for dl in doc_lens:
        prob = scores[idx:idx+dl]
        label = labels[idx:idx+dl]
        for aid in range(dl):
            output.write("%d 0 %d 0 %.6f 0\n" % (qid,aid,prob[aid]))
        qid += 1
        idx += dl
        if idx >= len(labels):
            break
    output.close()


def accuracy(labels,one_probs,doc_lens):
    idx = 0
    correct = 0
    total = 0
    for dl in doc_lens:
        prob = one_probs[idx:idx+dl]
        label = labels[idx:idx+dl]
        sum_label = sum(label)
        if sum_label==0 or sum_label==dl:
            idx += dl
            continue
        off = prob.argmax()
        correct += labels[idx+off]
        idx += dl
        total += 1.0
    acc = 1.0 * correct / total
    return acc


def mrr_metric(labels,one_probs,doc_lens,data_type="training"):
    dump_trec_format(labels,one_probs,doc_lens)
    popen = sp.Popen(["../trec_eval/trec_eval",
                "-m", "recip_rank",
                os.path.join(FLAGS.preprocessed_data_directory,FLAGS.data_mode,data_type+".rel_info"),
                "temp.trec_res"],
                stdout=sp.PIPE)
    with popen.stdout as f:
        metric = f.read().strip("\n")[-6:]
        mrr = float(metric)
    return mrr


def map_score(labels,one_probs,doc_lens,data_type="training"):
    dump_trec_format(labels,one_probs,doc_lens)
    popen = sp.Popen(["../trec_eval/trec_eval",
                "-m", "map",
                os.path.join(FLAGS.preprocessed_data_directory,FLAGS.data_mode,data_type+".rel_info"),
                "temp.trec_res"],
                stdout=sp.PIPE)
    with popen.stdout as f:
        metric = f.read().strip("\n")[-6:]
        map_sc = float(metric)
    return map_sc


def load_trec_prediction(filename):
    predictions = []
    sample_pred = []
    prev = 1
    for line in open(filename,'r'):
        line = line.strip("\n")
        if line=='': continue
        sid,_,aid,_,pred,_ = line.split()
        sid = int(sid)
        aid = int(aid)
        pred = float(pred)
        if prev != sid:
            # normalize predictions
            _sum = sum(sample_pred)
            sample_pred = [(1.0*x)/_sum if _sum!=0 else x for x in sample_pred]
            predictions.append(sample_pred)
            sample_pred = []
        prev = sid
        sample_pred.append(pred)
    _sum = sum(sample_pred)
    sample_pred = [(1.0*x)/_sum for x in sample_pred]
    predictions.append(sample_pred)
    return predictions


def get_sentence_lvl_dataset(probs,batch,wang_preds,mode,ign_indexes):
    prob_one = probs[:,:,0]
    labels = batch.labels[:,:,0]
    bs,ld = prob_one.shape
    
    cnt_score = batch.cnt_score
    isf_score = batch.isf_score
    idf_score = batch.idf_score
    locisf_score = batch.locisf_score
    sent_lens = batch.sent_lens # [bs,ld]

    loc_isf_score = batch.locisf_score
    cnt_score = batch.cnt_score

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

        if mode=='wikiqa':
            meta_x = np.vstack([
                prob_one[i,:doc_len],
                sent_lens[i,:doc_len],
                idf_score[i,:doc_len],
                isf_score[i,:doc_len],
                wang_preds[idx][:doc_len],
                ])
        elif mode=="newsqa":
            meta_x = np.vstack([
                prob_one[i,:doc_len],
                isf_score[i,:doc_len],
                locisf_score[i,:doc_len],
                wang_preds[idx][:doc_len],
                ])
        elif mode=="squad":
            meta_x = np.vstack([
                prob_one[i,:doc_len],
                idf_score[i,:doc_len],
                isf_score[i,:doc_len],
                wang_preds[idx][:doc_len],
                ])
        elif mode=="msmarco":
            meta_x = np.vstack([
                prob_one[i,:doc_len],
                cnt_score[i,:doc_len],
                isf_score[i,:doc_len],
                wang_preds[idx][:doc_len],
                ])
        X.append(meta_x)
        Y.extend(labels[i,:doc_len])
    X = np.hstack(X).T
    Y = np.array(Y,dtype=np.float32)

    return X,Y,filt_doc_len



