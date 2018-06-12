####################################
# Author: Ronald Cardenas
# Date: July 2017
# Project: Document Modeling with External Attention for Sentence Extraction

####################################

import os,sys
import pdb
import nltk
import cPickle as pickle
import hashlib
import json
import numpy as np
import io
import string
from wordvecs_authors import *
from nltk.corpus import stopwords as stw
from collections import defaultdict

np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # datasets/wikiqa
DATASETS_BASEDIR = os.path.dirname(BASE_DIR)
DATA_PICKLE_DIR = os.path.join(BASE_DIR,'wiki_code_pack')
WRD_EMB_DIR = os.path.join(DATASETS_BASEDIR,'word_emb')
GOLD_SUMM_DIR = os.path.join(DATASETS_BASEDIR,'gold_folders')
PREPROC_DATA_DIR = os.path.join(DATASETS_BASEDIR,'preprocessed_data')
#########################
WRD_EMB_MODEL = '1-billion-word-language-modeling-benchmark-r13output.word2vec.vec' # trained on 1BW

# Special IDs
PAD_ID = 0
UNK_ID = 1

# UNK THRESHOLD - exclusive
THR = 1

#####################################################################
def saveObject(obj, name='model'):
    with open(name + '.pickle', 'wb') as fd:
        pickle.dump(obj, fd, protocol=pickle.HIGHEST_PROTOCOL)


def uploadObject(obj_name,alt_format=False):
    # Load tagger
    obj = ''
    if not alt_format:
        with open(obj_name + '.pickle', 'rb') as fd:
            obj = pickle.load(fd)
    else:
        with open(obj_name + '.pkl', 'rb') as fd:
            obj = pickle.load(fd)
    return obj

def get_stopwords_ids(vocab):
    stopwords = stw.words('english')
    sw_ids = [vocab[x] for x in stopwords if x in vocab]
    return sw_ids

def get_short_stpw_ids(vocab):
    raw = open("short-stopwords.txt",'r').read().strip("\n").split("\n")
    ids = [vocab[x.lower()] for x in raw if x in vocab]
    return ids

def get_short_stpw():
    raw = open("short-stopwords.txt",'r').read().strip("\n").split("\n")
    raw = [x.lower() for x in raw]
    return raw


def read_data(split,get_feats=False):
    data_idx = 1
    if split == 'training':
        data_idx = 1
    elif split == 'validation':
        data_idx = 2
    else:
        data_idx = 3

    if split=="test":
        return read_unfiltered_test(get_feats)

    #data,wvecs,max_sent_len = uploadObject(os.path.join(DATA_PICKLE_DIR,'wiki_cnn'),True)
    data,wvecs,max_sent_len = uploadObject(os.path.join(DATA_PICKLE_DIR,'wiki_cnn_filt'),True)
    
    sentById   = {}
    labelsById = {}
    questionsById = {}
    featsById = {}
    prev_sid,prev_qid = -1,-1
    count = 0
    sum_labels = {}

    if split=="training":
        for item in data:
            if item["split"] != data_idx:
                continue
            qid = item["qid"]-1
            if qid not in sum_labels:
                sum_labels[qid] = 0
            sum_labels[qid] += item["y"]

    for item in data:
        if item["split"] != data_idx:
            continue
        qid = item["qid"]-1
        sid = item["aid"]
        if prev_qid!=-1 and prev_qid==qid and sid <= prev_sid:
            pdb.set_trace()
        if data_idx==1:
            if sum_labels[qid]==0:
                continue
        prev_qid = qid
        prev_sid = sid
        if qid not in sentById:
            sentById[qid] = []
            labelsById[qid] =[]
            featsById[qid] = []
            questionsById[qid] = item["question"].lower()
        sentById[qid].append(item["answer"].lower())
        labelsById[qid].append(item["y"])
        if get_feats:
            featsById[qid].append(item["features"])
        
        if count%10000 == 0:
            print "-->read_data_count:",count
        count +=1
    #END-FOR-DATA
    out = [sentById,questionsById,labelsById]
    if get_feats:
        out.append(featsById)

    return out

'''
read from txt original unfiltered txt files
get_not_ans: allows to retrieve not answered samples too (for Wang's vocab building)
'''
def read_from_txt(split,get_feats,get_not_ans=False):
    data_idx = 1
    fn = ''
    if split == 'training':
        data_idx = 1
        fn = "WikiQACorpus/WikiQA-train.txt"
    elif split == 'validation':
        data_idx = 2
        fn = "WikiQACorpus/WikiQA-dev.txt"
    else:
        data_idx = 3
        fn = "WikiQACorpus/WikiQA-test.txt"
    data,wvecs,max_sent_len = uploadObject(os.path.join(DATA_PICKLE_DIR,'wiki_cnn_filt'),True)
    
    sentById   = {}
    labelsById = {}
    questionsById = {}
    featsById = {}
    prev_q = "-"
    cand = []
    curr_label = []
    
    sum_labels = {}
    data = [item for item in data if item["split"] == data_idx]
    for item in data:
        qid = item["qid"]-1
        if qid not in sum_labels:
            sum_labels[qid] = 0
        sum_labels[qid] += item["y"]
    qid = 0
    for line in open(fn,'r'):
        line = line.strip("\n")
        if line=='': continue
        qst,content,label = line.lower().split('\t')

        label = int(label)
        if prev_q!='-' and prev_q!=qst:
            if sum(curr_label)!=0 or get_not_ans:
                questionsById[qid] = prev_q
                sentById[qid] = cand
                labelsById[qid] = curr_label
                qid += 1
            cand = []
            curr_label = []
            
        cand.append(content)
        curr_label.append(label)
        prev_q = qst
    ##
    if sum(curr_label)!=0 or get_not_ans:
        questionsById[qid] = prev_q
        sentById[qid] = cand
        labelsById[qid] = curr_label

    out = [sentById,questionsById,labelsById]
    if get_feats:
        for _id,qst in questionsById.items():
            featsById[_id] = []
            for item in data:
                if qst == item["question"].lower():
                    featsById[_id].append(item["features"])
        ##
        out.append(featsById)
    return out



def read_unfiltered_test(get_feats):
    fn = "WikiQACorpus/WikiQA-test.txt"
    prev_q = "-"
    qid = 0
    questions,conts,labels = {},{},{}
    cand = []
    curr_label = []
    for line in open(fn,'r'):
        line = line.strip("\n")
        if line=='': continue
        qst,content,label = line.lower().split('\t')
        label = int(label)
        if prev_q!='-' and prev_q!=qst:
            if sum(curr_label)!=0:
                questions[qid] = prev_q
                conts[qid] = cand
                labels[qid] = curr_label
                qid += 1
            cand = []
            curr_label = []
            
        cand.append(content)
        curr_label.append(label)
        prev_q = qst
    ##
    if sum(curr_label)!=0:
        questions[qid] = prev_q
        conts[qid] = cand
        labels[qid] = curr_label
    if get_feats:
        return [conts,questions,labels,'']
    else:
        return [conts,questions,labels]



def get_modified_vocab(anonymized_setting=False,force=False):
    '''
    @param anonymized_setting: True if data is anonymized (default False)
    @param force: True to rebuild the vocabulary and save the object
    reads vocabulary from word embeddings models and adds PADDING and UNK tokens.
          saves vocab object if not present or forced to
    '''
    # Read word embedding file
    wordembed_filename = os.path.join(WRD_EMB_DIR,WRD_EMB_MODEL)
    vocab_name = os.path.join(PREPROC_DATA_DIR,'wikiqa','vocab_org')
    vocab_dict = {}
    if os.path.exists(vocab_name+'.pickle') and not force:
        vocab_dict = uploadObject(vocab_name)
        return vocab_dict
    print "  Using Word Emb model:"
    print "  %s" % wordembed_filename
    ### Building vocabulary
    # Add padding
    vocab_dict["_PAD"] = PAD_ID
    # Add UNK
    vocab_dict["_UNK"] = UNK_ID
    linecount = 0
    with open(wordembed_filename, "r") as fembedd:
        for line in fembedd:
            if linecount != 0:
                linedata = line.split()
                vocab_dict[linedata[0]] = linecount + 1
            if linecount%10000 == 0:
                print(str(linecount)+" ...")
            linecount += 1
    print("Size of vocab: %d (_PAD:0, _UNK:1)"%len(vocab_dict))
    # save fresh object
    saveObject(vocab_dict,vocab_name)
    return vocab_dict


def get_local_vocab(force=False):
    """
    Extracted from WikiQA whole dataset
    """
    vocab_name = os.path.join(PREPROC_DATA_DIR,'wikiqa','vocab_local_org')

    if os.path.exists(vocab_name+'.pickle') and not force:
      vocab_dict = uploadObject(vocab_name)
      return vocab_dict
    vocab_dict = {}
    freq_dict = {}
    # Add padding
    vocab_dict["_PAD"] = PAD_ID
    # Add UNK
    vocab_dict["_UNK"] = UNK_ID
    count = 0
    for csplit in ["training","validation","test"]:
        sentById,questionsById,labelsById = read_data(csplit)
        for _id,question in questionsById.items():
          sents = sentById[_id]
          sents.append(question)
          for sent in sents:
            words = sent.split()
            for w in words:
              if w in vocab_dict:
                continue
              vocab_dict[w] = count + 2
              count += 1
              if count%1000 == 0:
                print(str(count)+" - local vocab ...")
    
    print("Size of vocab: %d (_PAD:0, _UNK:1)" % len(vocab_dict))
    saveObject(vocab_dict,vocab_name)

    return vocab_dict


def get_hash(question):
    story_id = ''.join(np.random.choice(list(string.ascii_uppercase + string.digits),len(question)) )
    return hashlib.md5(story_id.encode('utf-8') + "::" + question.encode('utf-8')).hexdigest()


def words_to_id(sentences,vocab_dict={}):
    new_sents = []
    for sent in sentences:
        sent = sent.split()
        word_ids = [vocab_dict[word] if word in vocab_dict else UNK_ID for word in sent]
        new_sents.append(word_ids)
    return new_sents


def get_idf_baseline_dict(stopwords,force=False):
    idf_fn = os.path.join(PREPROC_DATA_DIR,"wikiqa","idf_base_score_dict")
    if os.path.exists(idf_fn + ".pickle") and not force:
        idf_dict = uploadObject(idf_fn)
        return idf_dict

    ndocs = 0
    idf_dict = defaultdict(float)
    for csplit in ["training","validation","test"]:
        sentById,questionsById,labelsById = read_data(csplit)
        for _id,question in questionsById.items():
            ndocs += 1
            qset = set(question.split())
            for w in qset:
                if w in stopwords: continue
                idf_dict[w] += 1.0
            if ndocs%500 == 0:
                print "-->idf_base_dict_count:",ndocs
    for w in idf_dict:
        idf_dict[w] = np.log(ndocs / idf_dict[w])

    saveObject(idf_dict,idf_fn)
    return idf_dict



def get_isf_idf_dict(vocab,force=False):
    isf_fn = os.path.join(PREPROC_DATA_DIR,"wikiqa","isf_score_dict")
    idf_fn = os.path.join(PREPROC_DATA_DIR,"wikiqa","idf_score_dict")
    if os.path.exists(isf_fn + ".pickle") and os.path.exists(idf_fn + ".pickle") and not force:
        isf_dict = uploadObject(isf_fn)
        idf_dict = uploadObject(idf_fn)
        return isf_dict,idf_dict

    total_counts = nltk.FreqDist()
    total_counts_doc = nltk.FreqDist()
    nsents = 0
    ndocs = 0
    count = 0
    
    sentById,questionsById,labelsById = read_data("training")
    for _id,question in questionsById.items():
        sents = sentById[_id]
        labels = labelsById[_id]
        nsents += len(sents)
        ndocs += 1
        ref_sents = words_to_id(sents,vocab)
        doc_set = set()
        for sent in ref_sents:
            total_counts.update(set(sent))
            doc_set.update(sent)
        total_counts_doc.update(doc_set)
        if count%500 == 0:
            print "-->isf_dict_count:",count
        count +=1
    isf_dict = {}
    idf_dict = {}
    for wid,freq in total_counts.items():
        isf_dict[wid] = isf_score(nsents,freq)
    for wid,freq in total_counts_doc.items():
        idf_dict[wid] = isf_score(ndocs,freq)
    saveObject(isf_dict,isf_fn)
    saveObject(idf_dict,idf_fn)
    return isf_dict,idf_dict


def isf_score(N,nd):
    return np.log(1.0*N/nd + 1)


def eval_cnts(question,sents,isf_dict,idf_dict,stopwords):
    '''
    question: [wids] of queston
    sents: [... [wids]] for every sent
    '''
    question_set = set(question)

    count_scores = []
    idf_scores = []
    isf_scores = []
    local_isf_scores = []
    
    local_cnt_dict = nltk.FreqDist()
    n_sents = len(sents)
    for sent in sents:
        local_cnt_dict.update(set(sent))
    
    for sent in sents:
        sent_set = set(sent)
        _isf_score = idf_score = local_isf_score = 0.0
        cnt_sc = 0
        for wid in question_set:
            if wid in sent_set and wid not in stopwords:
                if wid in isf_dict:
                    _isf_score += isf_dict[wid]
                else:
                    _isf_score += isf_dict[UNK_ID]
                if wid in idf_dict:
                    idf_score += idf_dict[wid]
                    cnt_sc += 1
                else:
                    idf_score += idf_dict[UNK_ID]
                    cnt_sc += 1
                local_isf_score += isf_score(n_sents,local_cnt_dict[wid])
        idf_scores.append(idf_score)
        isf_scores.append(_isf_score)
        local_isf_scores.append(local_isf_score)
        count_scores.append(cnt_sc)
        
    return isf_scores,idf_scores,local_isf_scores,count_scores


def eval_cnts_base(question,sents,idfbase_dict,stopwords):
    '''
    Evals Cnt & Wgt Cnt
    '''
    question_set = set(question.split())
    cnt_scores = []
    wgt_scores = []

    for sent in sents:
        sent_set = set(sent.split())
        cnt = 0
        wgt_score = 0.0
        for word in question_set:
            if word in sent_set and word not in stopwords:
                cnt += 1
                wgt_score += idfbase_dict[word]
        if cnt==0:
            print("-",question)
            print("-",sent)
            pdb.set_trace()
        cnt_scores.append(cnt)
        wgt_scores.append(wgt_score)
    return cnt_scores,wgt_scores

