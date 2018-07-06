####################################
# Author: Shashi Narayan
# Date: September 2016
# Project: Document Summarization
# H2020 Summa Project

# v1.2 XNET
#   author: Ronald Cardenas
####################################

"""
Question Answering Modules and Models
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('../../common')

import numpy as np
import tensorflow as tf
import random
import os
import pdb
try:
    import cPickle as pickle
except:
    import pickle

from my_flags import FLAGS
from model_utils import convert_logits_to_softmax
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

def write_prediction_summaries(batch, pred_probs, modelname, data_type):
    print("Writing predictions and final summaries ...")
    # Save Output Logits
    np.save(FLAGS.train_dir+"/"+modelname+"."+data_type+"-prediction", pred_probs)

    # Writing
    #write_predictions(batch, modelname+"."+data_type, pred_logits)

def write_cos_sim(cos_sim, modelname, data_type):
    print("Writing cos sim ...")
    np.save(FLAGS.train_dir+"/"+modelname+"."+data_type+"-cos_sim", cos_sim)

def load_prediction(modelname):
    logits = np.load(FLAGS.train_dir+"/" + modelname + '.npy')
    return logits

def write_predictions(batch,file_prefix, np_predictions):
    foutput = open(FLAGS.train_dir+"/"+file_prefix+".predictions", "w")
    np_labels = batch.labels[:,:,0]
    for fileindex,filename in enumerate(batch.docnames):
        foutput.write(filename+"\n")

        sentcount = 0
        for sentpred, sentlabel in zip(np_predictions[fileindex], np_labels[fileindex]):
            one_prob = sentpred[0]  # <-------------- ISN'T INDEX 0 THE PROB OF C==0 | nope it's prob of 1
            label = sentlabel[0]

            if self.weights[fileindex][sentcount] == 1:
                foutput.write(str(int(label))+"\t"+str(one_prob)+"\n")
            else:
                break
            sentcount += 1
        foutput.write("\n")
    foutput.close()

#####################################################################
class BatchData(object):
    def __init__(self,docnames,docs,labels,weights,isf,isf_id,idf,locisf):
        self.docnames = docnames
        self.docs = docs
        self.labels = labels
        self.weights = weights
        self.isf_score = isf
        self.idf_score = idf
        self.isf_score_ids = isf_id
        self.locisf_score = locisf
        self.logits = None
        self.cos_sim = None
        self.initial_extend = True # False once start expanding the batch

    def extend(self,batch):
        if self.initial_extend:
            self.docnames = []
            self.docs = []
            self.labels = []
            self.weights = []
            self.isf_score = []
            self.idf_score = []
            self.isf_score_ids = []
            self.locisf_score = []
            self.cos_sim = []
            self.logits = []
            self.initial_extend = False
        self.docnames.append(batch.docnames)
        self.docs.append(batch.docs)
        self.labels.append(batch.labels)
        self.cos_sim.append(batch.cos_sim)
        self.weights.append(batch.weights)
        self.isf_score.append(batch.isf_score)
        self.idf_score.append(batch.idf_score)
        self.isf_score_ids.append(batch.isf_score_ids)
        self.locisf_score.append(batch.locisf_score)
        self.logits.append(batch.logits)
        
    def concat_batches(self):
        if self.logits[0] != None:
            self.logits = np.vstack(self.logits)
        if self.cos_sim[0]!=None:
            self.cos_sim = np.vstack(self.cos_sim)
        self.docs = np.vstack(self.docs)
        self.labels = np.vstack(self.labels)
        self.weights = np.vstack(self.weights)
        self.isf_score = np.vstack(self.isf_score)
        self.idf_score = np.vstack(self.idf_score)
        self.isf_score_ids = np.vstack(self.isf_score_ids)
        self.locisf_score = np.vstack(self.locisf_score)


class Data(object):
    def __init__(self, vocab_dict, data_type, normalizer=None, pca_model=None):
        self.filenames = []
        self.docs = []
        self.titles = []
        self.labels = []
        self.isf_scores = []
        self.idf_scores = []
        self.locisf_scores= []
        self.sorted_isf_score_indexes = []
        self.weights = []
        self.fileindices = []
        self.normalizer = normalizer
        self.pca_model = pca_model

        self.data_type = data_type

        # populate the data
        self.populate_data(vocab_dict, data_type)


    def get_batch(self, startidx, endidx):
        # This is very fast if you keep everything in Numpy

        # Numpy dtype
        dtype = np.float16 if FLAGS.use_fp16 else np.float32

        # For train, (endidx-startidx)=FLAGS.batch_size, for others its as specified
        batch_docnames = np.empty((endidx-startidx), dtype="S40") # File ID of size 40
        batch_docs = np.empty(((endidx-startidx), (FLAGS.max_doc_length + FLAGS.max_title_length + FLAGS.max_image_length +
                                                   FLAGS.max_firstsentences_length + FLAGS.max_randomsentences_length), FLAGS.max_sent_length), dtype="int32")
        batch_label  = np.empty(((endidx-startidx), FLAGS.max_doc_length, FLAGS.target_label_size), dtype=dtype)
        batch_weight = np.empty(((endidx-startidx), FLAGS.max_doc_length), dtype=dtype)
        batch_isf_score_ids  = np.empty(((endidx-startidx), FLAGS.topK), dtype=np.int32)
        
        batch_isf_score  = np.empty(((endidx-startidx), FLAGS.max_doc_length), dtype=dtype)        
        batch_idf_score  = np.empty(((endidx-startidx), FLAGS.max_doc_length), dtype=dtype)
        batch_locisf_score  = np.empty(((endidx-startidx), FLAGS.max_doc_length), dtype=dtype)

        batch_idx = 0
        
        for fileindex in self.fileindices[startidx:endidx]:
            # Document Names
            batch_docnames[batch_idx] = self.filenames[fileindex][67:-14]

            # Document
            doc_wordids = self.docs[fileindex][:] # [FLAGS.max_doc_length, FLAGS.max_sent_length]
            doc_wordids = [self.process_to_chop_pad(thissent, FLAGS.max_sent_length) for thissent in doc_wordids] # update sentence len
            doc_wordids = doc_wordids[:FLAGS.max_doc_length] # update doc len
            doc_wordids = doc_wordids + [self.process_to_chop_pad([], FLAGS.max_sent_length)]*(FLAGS.max_doc_length - len(doc_wordids))

            if (FLAGS.max_title_length > 0):
                title_sents = [self.process_to_chop_pad(thissent, FLAGS.max_sent_length) for thissent in self.titles[fileindex]]
                title_sents = title_sents[:FLAGS.max_title_length]
                title_sents = title_sents + [self.process_to_chop_pad([], FLAGS.max_sent_length)]*(FLAGS.max_title_length - len(title_sents))
                doc_wordids = doc_wordids + title_sents # [FLAGS.max_title_length, FLAGS.max_sent_length]
            
            batch_docs[batch_idx] = np.array(doc_wordids[:], dtype="int32")

            # Labels
            labels = self.labels[fileindex][:FLAGS.max_doc_length]
            labels = labels + [0]*(FLAGS.max_doc_length - len(labels))
            # labels: (max_doc_length) --> labels_vecs: (max_doc_length, target_label_size)
            labels_vecs = [[1, 0] if (label==1) else [0, 1] for label in labels]
            batch_label[batch_idx] = np.array(labels_vecs[:], dtype=dtype)

            # Weights
            weights = self.weights[fileindex][:FLAGS.max_doc_length]
            weights = weights + [0]*(FLAGS.max_doc_length - len(weights))
            batch_weight[batch_idx] = np.array(weights[:], dtype=dtype)

            # ISF Score ids
            isf_score_ids = self.sorted_isf_score_indexes[fileindex][:FLAGS.topK]
            isf_score_ids = isf_score_ids + [-1]*(FLAGS.topK - len(isf_score_ids))
            batch_isf_score_ids[batch_idx] = np.array(isf_score_ids[:],dtype=np.int32)

            # ISF scores
            
            isf_sc = self.isf_scores[fileindex][:FLAGS.max_doc_length]
            isf_sc = isf_sc + [0]*(FLAGS.max_doc_length - len(isf_sc))
            batch_isf_score[batch_idx] = np.array(isf_sc[:],dtype=dtype)

            # IDF scores
            idf_sc = self.idf_scores[fileindex][:FLAGS.max_doc_length]
            idf_sc = idf_sc + [0]*(FLAGS.max_doc_length - len(idf_sc))
            batch_idf_score[batch_idx] = np.array(idf_sc[:],dtype=dtype)

            # Local ISF scores

            locisf_sc = self.locisf_scores[fileindex][:FLAGS.max_doc_length]
            locisf_sc = locisf_sc + [0]*(FLAGS.max_doc_length - len(locisf_sc))
            batch_locisf_score[batch_idx] = np.array(locisf_sc[:],dtype=dtype)

            # increase batch count
            batch_idx += 1
        #END-FOR-FILEIDX

        batch = BatchData(  docnames= batch_docnames,
                            docs    = batch_docs,
                            labels  = batch_label,
                            weights = batch_weight,
                            isf     = batch_isf_score,
                            isf_id  = batch_isf_score_ids,
                            idf     = batch_idf_score,
                            locisf  = batch_locisf_score)

        return batch

    def shuffle_fileindices(self):
        random.shuffle(self.fileindices)

    def process_to_chop_pad(self, orgids, requiredsize):
        if (len(orgids) >= requiredsize):
            return orgids[:requiredsize]
        else:
            padids = [PAD_ID] * (requiredsize - len(orgids))
            return (orgids + padids)

    def populate_data(self, vocab_dict, data_type):      
        
        full_data_file_prefix = ""
        label_prefix = ""
        scores_file_prefix = ""
        full_data_file_prefix = FLAGS.preprocessed_data_directory + "/" + FLAGS.data_mode + "/" + data_type + '.org_ent'
        scores_file_prefix = FLAGS.preprocessed_data_directory + "/" + FLAGS.data_mode + "/" + data_type

        print("Data file prefix (.doc, .question, .label, .score): %s" % full_data_file_prefix)
        
        # Process doc, title, image and label
        doc_data_list = open(full_data_file_prefix+".doc",'r').read().strip().split("\n\n")
        title_data_list = open(full_data_file_prefix+".question",'r').read().strip().split("\n\n")
        label_data_list = open(full_data_file_prefix+".label",'r').read().strip().split("\n\n") # Use collective oracle
        
        isf_scores_data_list = open(scores_file_prefix+".isf.scores",'r').read().strip().split("\n\n") # ISF scores for each sentence
        if FLAGS.use_idf:
            idf_scores_data_list = open(scores_file_prefix+".idf.scores",'r').read().strip().split("\n\n") # ISF scores for each sentence
        if FLAGS.use_locisf:
            locisf_scores_data_list = open(scores_file_prefix+".locisf.scores",'r').read().strip().split("\n\n") # Local ISF scores for each sentence
        
        #image_data_list = open(full_data_file_prefix+".paraphr").read().strip().split("\n\n")

        # insert here file init for query paraphrase data
        print("Data sizes: %d %d %d"%(len(doc_data_list), len(title_data_list), len(label_data_list) ))

        print("Preparing data based on model requirement ...")
        doccount = 0
        ndata = len(doc_data_list)
        iter_indexes = range(ndata)

        extra_features = [] # [total_sentences, #n_extra_feats (3 so far)]
        n_features = (FLAGS.use_locisf + FLAGS.use_isf + FLAGS.use_idf)

        if data_type != 'test' and FLAGS.use_subsampled_dataset:
            fn = FLAGS.preprocessed_data_directory + "/" + FLAGS.data_mode + "/"+data_type+".subsampling_indexes"
            iter_indexes = [int(x) for x in open(fn,'r').read().strip().split("\n")]
            print("Subsampled data size: ",len(iter_indexes))

        for doc_idx in iter_indexes:
            doc_data   = doc_data_list[doc_idx]
            title_data = title_data_list[doc_idx]
            label_data = label_data_list[doc_idx]
            isf_data = isf_scores_data_list[doc_idx]
            
            if FLAGS.use_idf:
                idf_data = idf_scores_data_list[doc_idx]
                idf_lines = idf_data.strip().split("\n")
            if FLAGS.use_locisf:
                locisf_data = locisf_scores_data_list[doc_idx]
                locisf_lines = locisf_data.strip().split("\n")
            

            doc_lines   = doc_data.strip().split("\n")
            title_lines = title_data.strip().split("\n")
            label_lines = label_data.strip().split("\n")
            isf_lines = isf_data.strip().split("\n")
            
            filename = doc_lines[0].strip()

            if ((filename == title_lines[0].strip()) and (filename == label_lines[0].strip())):
                # Put filename
                self.filenames.append(filename)
                # Doc & sent_lens
                thisdoc = []
                doc_len = min(len(doc_lines)-1,FLAGS.max_doc_length)
                for idx in range(doc_len):
                    thissent = [int(item) for item in doc_lines[idx+1].strip().split()]
                    thissent = thissent[:FLAGS.max_sent_length]
                    thisdoc.append(thissent)
                self.docs.append(thisdoc)

                # Title
                thistitle = []
                for idx in range(min(len(title_lines)-1,FLAGS.max_title_length)):
                    thissent = [int(item) for item in title_lines[idx+1].strip().split()]
                    thissent = thissent[:FLAGS.max_sent_length]
                    thistitle.append(thissent)
                self.titles.append(thistitle)

                # Labels 1/0, 1, 0 and 2 -> 0 || Weights
                thislabel = []
                thisweight = []
                # Scores
                this_isf = []
                this_locisf = []
                this_idf = []
                this_isf_ids = []
                doc_scores = np.zeros([doc_len,n_features],dtype=np.float32)
                for idx in range(doc_len):
                    thissent_label = int(label_lines[idx+1].strip())
                    thissent_weight = 1
                    isf = float(isf_lines[idx+1])
                    sort_idx = idx
                    if FLAGS.use_locisf:
                        locisf = float(locisf_lines[idx+1])
                        this_locisf.append(locisf)
                    if FLAGS.use_idf:
                        idf = float(idf_lines[idx+1])
                        this_idf.append(idf)
                    thislabel.append(thissent_label)
                    thisweight.append(thissent_weight)
                    this_isf.append(isf)
                    this_isf_ids.append( (sort_idx,isf) )
                self.labels.append(thislabel)
                self.weights.append(thisweight)
                this_isf_ids.sort(reverse=True,key=lambda x: x[1])
                self.sorted_isf_score_indexes.append([x[0] for x in this_isf_ids])
                # fill in scores
                idx = 0
                if FLAGS.use_isf:
                    doc_scores[:,idx] = this_isf
                    idx += 1
                if FLAGS.use_idf:
                    doc_scores[:,idx] = this_idf
                    idx += 1
                if FLAGS.use_locisf:
                    doc_scores[:,idx] = this_locisf
                extra_features.append(doc_scores)
                
            #END-IF
            else:
                print("Some problem with %s.* files. Exiting!" % full_data_file_prefix)
                exit(0)

            if doccount%10000==0:
                print("%d ..."%doccount)
            doccount += 1
        #END-FOR-DATA
        self.fileindices = list(range(len(self.filenames)))
        extra_features = np.vstack(extra_features)
        if FLAGS.norm_extra_feats:
            print("Normalizing extra features (z-score)...")
            if data_type=='training':
                # define Standarizer
                self.normalizer = StandardScaler()
                self.normalizer.fit(extra_features)
            extra_features = self.normalizer.transform(extra_features)

            # only decorrelate if normalized
            if FLAGS.decorrelate_extra_feats:
                print("Decorrelating extra features (PCA)...")
                if data_type=='training':
                    # define PCA model
                    self.pca_model = PCA(n_components=n_features-1,whiten=True)
                    self.pca_model.fit(extra_features)
                extra_features = self.pca_model.transform(extra_features)
        #END-NORM-DECORR

        # fill in extra features in Data object
        index = 0
        doccount = 0
        for doc in self.docs:
            doc_len = len(doc)
            idx = 0
            if FLAGS.use_isf:
                this_isf    = list(extra_features[index:index+doc_len,idx])
                self.isf_scores.append(this_isf)
                idx += 1
            if FLAGS.use_idf:
                this_idf    = list(extra_features[index:index+doc_len,idx])
                self.idf_scores.append(this_idf)
                idx += 1
            if FLAGS.use_locisf:
                this_locisf = list(extra_features[index:index+doc_len,idx])
                self.locisf_scores.append(this_locisf)
            index += doc_len
            if doccount%10000==0:
                print("%d ....."%doccount)
            doccount += 1
        #END-2nd-FOR-DOCS

class DataProcessor:
    def prepare_news_data(self, vocab_dict, data_type="training",normalizer=None,pca_model=None):
        data_obj_fn = ''
        if data_type != 'test' and FLAGS.use_subsampled_dataset:
            data_obj_fn = os.path.join(FLAGS.train_dir,data_type+"_subsampled")
        else:
            data_obj_fn = os.path.join(FLAGS.train_dir,data_type)

        if os.path.exists(data_obj_fn+'.pickle') and not FLAGS.force_reading:
            data = uploadObject(data_obj_fn)
        else:
            data = Data(vocab_dict,data_type,normalizer=normalizer,pca_model=pca_model)
            saveObject(data,data_obj_fn)
        return data

    def prepare_vocab_embeddingdict(self):
        vocab_fn = os.path.join(FLAGS.train_dir,'vocab-org')
        wde_fn = os.path.join(FLAGS.train_dir,'wde-org')
        if os.path.exists(vocab_fn+'.pickle'):
            vocab_dict = uploadObject(vocab_fn)
            word_embedding_array = uploadObject(wde_fn)
            return vocab_dict,word_embedding_array
        ####################################
        # Numpy dtype
        dtype = np.float16 if FLAGS.use_fp16 else np.float32

        vocab_dict = {}
        word_embedding_array = []

        # Add padding
        vocab_dict["_PAD"] = PAD_ID
        # Add UNK
        vocab_dict["_UNK"] = UNK_ID

        # Read word embedding file
        wordembed_filename = ""
        if FLAGS.anonymized_setting:
            wordembed_filename = FLAGS.pretrained_wordembedding_anonymdata
        else:
            wordembed_filename = FLAGS.pretrained_wordembedding_orgdata
        print("Reading pretrained word embeddings file: %s"%wordembed_filename)

        embed_line = ""
        linecount = 0
        with open(wordembed_filename, "r") as fembedd:
            for line in fembedd:
                if linecount == 0:
                    vocabsize = int(line.split()[0])
                    # Initiate fixed size empty array
                    word_embedding_array = np.empty((vocabsize, FLAGS.wordembed_size), dtype=dtype)
                else:
                    linedata = line.split()
                    vocab_dict[linedata[0]] = linecount + 1
                    embeddata = [float(item) for item in linedata[1:]][0:FLAGS.wordembed_size]
                    word_embedding_array[linecount-1] = np.array(embeddata, dtype=dtype)

                if linecount%10000 == 0:
                    print(str(linecount)+" ...")
                linecount += 1
        print("Read pretrained embeddings: %s"%str(word_embedding_array.shape))

        print("Size of vocab: %d (_PAD:0, _UNK:1)"%len(vocab_dict))
        vocabfilename = ""
        if FLAGS.anonymized_setting:
            vocabfilename = FLAGS.train_dir+"/vocab-anonym"
        else:
            vocabfilename = FLAGS.train_dir+"/vocab-org"
        print("Writing vocab file: %s"%vocabfilename)
        foutput = open(vocabfilename,"w")
        vocab_list = [(vocab_dict[key], key) for key in vocab_dict.keys()]
        vocab_list.sort()
        vocab_list = [item[1] for item in vocab_list]
        foutput.write("\n".join(vocab_list)+"\n")
        foutput.close()

        saveObject(vocab_dict,vocab_fn)
        saveObject(word_embedding_array,wde_fn)
        return vocab_dict, word_embedding_array
