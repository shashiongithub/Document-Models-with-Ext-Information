####################################
# Author: Ronald Cardenas
# Date: July 2017
# Project: Document Modeling with External Attention for Sentence Extraction

####################################

import os,sys
import pandas as pd
import pdb
import nltk
import cPickle as pickle
import hashlib
import numpy as np
import io
from nltk.corpus import stopwords as stw
from sample import Sample

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # datasets/newsqa
DATASETS_BASEDIR = os.path.dirname(BASE_DIR)
DATA_CSV_DIR = os.path.join(BASE_DIR,'split_data')
MALUUBA_DIR = os.path.join(BASE_DIR,'maluuba','newsqa')
WRD_EMB_DIR = os.path.join(DATASETS_BASEDIR,'word_emb')
GOLD_SUMM_DIR = os.path.join(BASE_DIR,'gold_folders')
PREPROC_DATA_DIR = os.path.join(DATASETS_BASEDIR,'preprocessed_data')
#########################
WRD_EMB_MODEL = '1-billion-word-language-modeling-benchmark-r13output.word2vec.vec' # trained on 1BW
#WRD_EMB_MODEL = 'story-question_Org.txt.word2vec.vec' # trained on NewsQA
#########################

# Special IDs
PAD_ID = 0
UNK_ID = 1

# Top K sentences scored by ISF

#####################################################################
def saveObject(obj, name='model'):
    with open(name + '.pickle', 'wb') as fd:
        pickle.dump(obj, fd, protocol=pickle.HIGHEST_PROTOCOL)


def uploadObject(obj_name):
    # Load tagger
    with open(obj_name + '.pickle', 'rb') as fd:
        obj = pickle.load(fd)
    return obj

#####################################################################

def read_data(prefix='training'):
    '''
    @param prefix: data csv file name [train,dev,test]
    returns a generator of tuples (story_id, [sent_tokenized_story], question)
    '''
    if prefix=='training':
        prefix='train'
    elif prefix=='validation':
        prefix='dev'
    data_dict = pd.read_csv(os.path.join(DATA_CSV_DIR,prefix+'.csv'),encoding='utf-8')
    nsample = len(data_dict)
    label_list = []
    sample_dict = {}
    for sid,content,question,tok_rngs in zip(data_dict["story_id"], \
                                            data_dict["story_text"], \
                                            data_dict["question"], \
                                            data_dict["answer_token_ranges"]):
        labels = get_labels(content,tok_rngs)
        if sum(labels)==0: # sub-task: at least one answer
            print("  skipped: ",sid,":: ",question)
            continue
        sample_id = get_hash(sid,question)
        if sample_id not in sample_dict:
            sample_dict[sample_id] = Sample(sample_id,content,question,labels)
        else:
            sample_dict[sample_id].labels += labels
            print(  "Repeated!!:: ", sid,":: ",question)
    return sample_dict
        


def get_labels(content,tok_rngs):
    rngs = []
    for str_rng in tok_rngs.split(','):
        temp = str_rng.split(':')
        l,r = int(temp[0]),int(temp[1])
        rngs.append( (l,r) )
    rngs.sort() # just in case, range lists look sorted
    acum_sent_len = [0]
    sents = nltk.sent_tokenize(content)
    for sent in sents:
        sent_len = len(sent.split())
        acum_sent_len.append( sent_len + acum_sent_len[-1] )
    nsents = len(acum_sent_len)-1
    labels  = np.zeros(nsents,dtype=int)
    for lb,rb in rngs:
        fst_sent = -1
        lst_sent = -1
        for i in xrange(1,nsents+1):
            if lb < acum_sent_len[i]:
                fst_sent = i-1
                break
        for i in xrange(fst_sent,nsents+1):
            if rb-1 < acum_sent_len[i]:
                lst_sent = i-1
                break
        labels[fst_sent:lst_sent+1] = 1
        if fst_sent>lst_sent:
            pdb.set_trace()

    return labels

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
        wordembed_filename = os.path.join(WRD_EMB_DIR,"story-question_Anonymized.txt.word2vec.vec")
        vocab_name = os.path.join(PREPROC_DATA_DIR,'newsqa','vocab_anonym')
    else:
        wordembed_filename = os.path.join(WRD_EMB_DIR,WRD_EMB_MODEL)
        vocab_name = os.path.join(PREPROC_DATA_DIR,'newsqa','vocab_org')
    vocab_dict = {}
    if os.path.exists(vocab_name+'.pickle') and not force:
        vocab_dict = uploadObject(vocab_name)
        return vocab_dict
    print("  Using Word Emb model:")
    print("  %s" % wordembed_filename)
    print("  Saving vocab in:")
    print("  %s" % vocab_name)
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
    isf_fn = os.path.join(PREPROC_DATA_DIR,"newsqa","isf_score_dict")
    idf_fn = os.path.join(PREPROC_DATA_DIR,"newsqa","idf_score_dict")
    if os.path.exists(isf_fn + ".pickle") and os.path.exists(idf_fn + ".pickle"):
        isf_dict = uploadObject(isf_fn)
        idf_dict = uploadObject(idf_fn)
        return isf_dict,idf_dict

    data_dict = pd.read_csv(os.path.join(DATA_CSV_DIR,'train.csv'),encoding='utf-8')

    sid_set = set()
    total_counts = nltk.FreqDist()
    total_counts_doc = nltk.FreqDist()
    nsents = 0
    ndocs = 0
    count = 0
    for sid,content in zip(data_dict["story_id"], data_dict["story_text"]):
        if sid in sid_set:
            continue
        sid_set.add(sid)
        sents = nltk.sent_tokenize(content)
        sents = [sent.split() for sent in sents] # sentences are already tokenized
        nsents += len(sents)
        ndocs += 1
        ref_sents = words_to_id(sents,vocab)
        doc_set = set()
        for sent in ref_sents:
            total_counts.update(set(sent))
            doc_set.update(sent)
        total_counts_doc.update(doc_set)
        if count%10000 == 0:
            print("-->isf_dict_count:",count)
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


####################################################################################

def trail_id(_id):
  sid = str(_id)
  sid = "0"*(7 - len(sid)) + sid
  return sid
