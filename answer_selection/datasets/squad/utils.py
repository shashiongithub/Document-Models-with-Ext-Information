####################################
# Author: Ronald Cardenas
# Date: July 2017
# Project: Document Modeling with External Attention for Sentence Extraction

####################################

import os,sys
import pandas as pd
import pdb
import nltk
import pickle
import hashlib
import json
import numpy as np
import io
from nltk.corpus import stopwords as stw
from sample import Sample

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # datasets/squad
DATASETS_BASEDIR = os.path.dirname(BASE_DIR)
DATA_JSON_DIR = os.path.join(BASE_DIR,'json_data')
WRD_EMB_DIR = os.path.join(DATASETS_BASEDIR,'word_emb')
GOLD_SUMM_DIR = os.path.join(DATASETS_BASEDIR,'gold_folders')
PREPROC_DATA_DIR = os.path.join(DATASETS_BASEDIR,'preprocessed_data')
#########################
WRD_EMB_MODEL = '1-billion-word-language-modeling-benchmark-r13output.word2vec.vec' # trained on 1BW

# Special IDs
PAD_ID = 0
UNK_ID = 1


#####################################################################
def saveObject(obj, name='model'):
    with open(name + '.pickle', 'wb') as fd:
        pickle.dump(obj, fd, protocol=pickle.HIGHEST_PROTOCOL)


def uploadObject(obj_name):
    # Load tagger
    with open(obj_name + '.pickle', 'rb') as fd:
        obj = pickle.load(fd)
    return obj


def get_labels(content,inits):
    rngs = []
    
    rngs.sort() # just in case, range lists look sorted
    acum_sent_len = [-1]
    sents = nltk.sent_tokenize(content)
    for sent in sents:
        acum_sent_len.append( len(sent) + acum_sent_len[-1] + 1) # adds blank space lost in sent_tokenization
    nsents = len(acum_sent_len)-1
    labels  = np.zeros(nsents,dtype=int)
    for lb in inits:
        sid = -1
        for i in xrange(1,nsents+1):
            if lb < acum_sent_len[i]:
                sid = i-1
                break
        labels[sid] = 1
        if sid==-1:
            pdb.set_trace()

    return labels


def read_data(split):
    filename = ''
    if split == 'training':
        filename = 'train-v1.1.json'
    else:
        filename = 'dev-v1.1.json'

    filename = os.path.join(DATA_JSON_DIR,filename)
    raw_data = json.load(open(filename,'r'))
    raw_data = raw_data["data"]

    data = []
    for article in raw_data:
        for paragraph in article["paragraphs"]:
            content = paragraph["context"]
            for qas in paragraph["qas"]:
                qid = qas["id"]
                question = qas["question"]
                char_rngs = []
                for ans in qas["answers"]:
                    char_rngs.append(ans["answer_start"])
                labels = get_labels(content,char_rngs)
                sid = get_hash(qid,question)
                sample = Sample(sid,content,question,labels)

                yield sample



def get_modified_vocab(anonymized_setting=False,force=False):
    '''
    @param anonymized_setting: True if data is anonymized (default False)
    @param force: True to rebuild the vocabulary and save the object
    reads vocabulary from word embeddings models and adds PADDING and UNK tokens.
          saves vocab object if not present or forced to
    '''
    # Read word embedding file
    wordembed_filename = ""
    vocab_fn = ''
    if anonymized_setting:
        wordembed_filename = os.path.join(WRD_EMB_DIR,WRD_EMB_MODEL)
        vocab_name = os.path.join(PREPROC_DATA_DIR,'squad','vocab_anonym')
    else:
        wordembed_filename = os.path.join(WRD_EMB_DIR,WRD_EMB_MODEL)
        vocab_name = os.path.join(PREPROC_DATA_DIR,'squad','vocab_org')
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


def get_hash(story_id,question):
    return hashlib.md5(story_id.encode('utf-8') + "::" + question.encode('utf-8')).hexdigest()


def get_stopwords_ids(vocab):
    stopwords = stw.words('english')
    sw_ids = [vocab[x] for x in stopwords if x in vocab]
    return sw_ids



def words_to_id(sentences,vocab_dict={}):
    new_sents = []
    for sent in sentences:
        word_ids = [vocab_dict[word] if word in vocab_dict else UNK_ID for word in sent]
        new_sents.append(word_ids)
    return new_sents


def get_isf_idf_dict(vocab):
    isf_fn = os.path.join(PREPROC_DATA_DIR,"squad","isf_score_dict")
    idf_fn = os.path.join(PREPROC_DATA_DIR,"squad","idf_score_dict")
    if os.path.exists(isf_fn + ".pickle") and os.path.exists(idf_fn + ".pickle"):
        isf_dict = uploadObject(isf_fn)
        idf_dict = uploadObject(idf_fn)
        return isf_dict,idf_dict

    data_gen = read_data("training")
    total_counts = nltk.FreqDist()
    total_counts_doc = nltk.FreqDist()
    nsents = 0
    ndocs = 0
    count = 0
    for sample in data_gen:
        sample_id,sents,question,labels = sample.unpack()
        ref_sents = words_to_id(sents,vocab)
        nsents += len(sents)
        ndocs += 1
        doc_set = set()
        for sent in ref_sents:
            total_counts.update(set(sent))
            doc_set.update(sent)
        total_counts_doc.update(doc_set)
        if count%10000 == 0:
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
    question = set(question)
    idf_scores = []
    cnt_scores = []
    isf_scores = []
    local_isf_scores = []
    local_cnt_dict = nltk.FreqDist()
    n_sents = len(sents)
    for sent in sents:
        local_cnt_dict.update(set(sent))

    for sent in sents:
        sent_set = set(sent)
        inters = [x for x in question.intersection(sent_set) if x not in stopwords]
        cnt_scores.append(len(inters))
        local_isf_score = sum([isf_score(n_sents,local_cnt_dict[wid]) for wid in inters])
        
        _isf_score = sum([isf_dict[wid] if wid in isf_dict else isf_dict[UNK_ID] for wid in inters])
        idf_score = sum([idf_dict[wid] if wid in idf_dict else idf_dict[UNK_ID] for wid in inters])
        idf_scores.append(idf_score)
        isf_scores.append(_isf_score)
        local_isf_scores.append(local_isf_score)
        
    return cnt_scores,isf_scores,idf_scores,local_isf_scores

