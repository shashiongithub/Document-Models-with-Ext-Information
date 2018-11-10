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


def accuracy(labels,one_probs,doc_lens,mode="first"):
    idx = 0
    correct = 0
    total = 0
    for dl in doc_lens:
        prob = one_probs[idx:idx+dl]
        label = labels[idx:idx+dl]
        if sum(label)==0:
            idx += dl
            continue

        if mode=="first":
            off = prob.argmax()
            correct += labels[idx+off]
        else:
            srt_ref = [(x,i) for i,x in enumerate(prob)]
            srt_ref.sort(reverse=True)
            correct += label[srt_ref[0][1]]

        idx += dl
        total += 1.0
    acc = 1.0 * correct / total
    return acc


def mrr_metric(labels,one_probs,doc_lens,data_type="training",mode="first"):
    return mrr_metric_local(labels,one_probs,doc_lens,data_type,mode)


def map_score(labels,one_probs,doc_lens,data_type="training",mode="first"):
    return map_score_local(labels,one_probs,doc_lens,data_type,mode)



###################################################################

def mrr_metric_trec(labels,one_probs,doc_lens,data_type="training"):
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


def mrr_metric_local(labels,one_probs,doc_lens,data_type="training",mode="first"):
    idx = 0
    mrr = 0.0
    total = 0.0
    for dl in doc_lens:
        probs = one_probs[idx:idx+dl].reshape([-1])
        label = labels[idx:idx+dl]
        if sum(label)==0:
            idx += dl
            continue

        srt_ref = []
        # tie breaking: earliest in list
        if mode=="first":
            srt_ref = [(-x,i) for i,x in enumerate(probs)]
            srt_ref.sort()
        # tie breaking: last in list
        else:
            srt_ref = [(x,i) for i,x in enumerate(probs)]
            srt_ref.sort(reverse=True)

        rel_rank = 0.0
        for idx_retr,(x,i) in enumerate(srt_ref):
            if label[i]==1:
                rel_rank = 1.0 + idx_retr
                break

        idx += dl
        mrr += 1.0/rel_rank # accumulate inverse rank
        total += 1.0
    mrr /= total
    return mrr


###################################################################

def map_score_trec(labels,one_probs,doc_lens,data_type="training",mode=""):
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


def map_score_local(labels,one_probs,doc_lens,data_type="training",mode="first"):
    idx = 0
    _map = 0.0
    total = 0.0
    for dl in doc_lens:
        probs = one_probs[idx:idx+dl].reshape([-1])
        label = labels[idx:idx+dl]
        if sum(label)==0:
            idx += dl
            continue
        srt_ref = []
        # tie breaking: earliest in list
        if mode=="first":
            srt_ref = [(-x,i) for i,x in enumerate(probs)]
            srt_ref.sort()
        # tie breaking: last in list
        else:
            srt_ref = [(x,i) for i,x in enumerate(probs)]
            srt_ref.sort(reverse=True)

        aps = 0.0
        n_corr = 0.0
        for idx_retr,(x,i) in enumerate(srt_ref):
            if label[i]==1:
                n_corr += 1.0
                aps += (n_corr / (idx_retr+1))
                # break
        aps /= sum(label)

        idx += dl
        total += 1.0
        _map += aps
    _map /= total
    return _map


###################################################################


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

