####################################
# Author: Ronald Cardenas
# Date: July 2017
# Project: Document Modeling with External Attention for Sentence Extraction

####################################

'''
Generates .doc and .question files, each with format
    <story_id>
    wid wid wid...
    wid wid ...

    <story_id>
    wid ...
    ...

'''
import os,sys
import pdb
import nltk
import io
from utils import *

if __name__ == "__main__":
    splits = ['training','validation','test']
    force = False
    anon_setting = False
    vocab = get_modified_vocab(anonymized_setting=anon_setting,force=force)
    stopwords = get_stopwords_ids(vocab)
    isf_dict,idf_dict = get_isf_idf_dict(vocab)

    for corpus_split in splits:
        mx_doc_len = 0
        mx_sent_len = 0
        data = read_data(corpus_split)
        # outputs files
        
        docs_out      = open(os.path.join(PREPROC_DATA_DIR,'newsqa',"%s.org_ent.doc"      % (corpus_split)), 'w')
        questions_out = open(os.path.join(PREPROC_DATA_DIR,'newsqa',"%s.org_ent.question" % (corpus_split)), 'w')
        labels_out    = open(os.path.join(PREPROC_DATA_DIR,'newsqa',"%s.org_ent.label"    % (corpus_split)), 'w')
        cnt_scores_out    = open(os.path.join(PREPROC_DATA_DIR,'newsqa',"%s.cnt.scores" % (corpus_split)), 'w')
        isf_scores_out    = open(os.path.join(PREPROC_DATA_DIR,'newsqa',"%s.isf.scores" % (corpus_split)), 'w')
        locisf_scores_out    = open(os.path.join(PREPROC_DATA_DIR,'newsqa',"%s.locisf.scores" % (corpus_split)), 'w')
        idf_scores_out    = open(os.path.join(PREPROC_DATA_DIR,'newsqa',"%s.idf.scores" % (corpus_split)), 'w')
        
        # write to output files
        count = 0
        nempties = 0
        for sample_id,sample in data.items():
            _,sents,question,labels = sample.unpack()
            labels = (labels>0).astype(float)
            fullpath_doc_name = os.path.join(DATASETS_BASEDIR,'newsqa',corpus_split,sample_id+'.summary.final')

            ref_sents = words_to_id(sents,vocab)
            ref_question = words_to_id([question],vocab)[0]

            cnt,isf,idf,locisf = eval_cnts(ref_question,ref_sents,isf_dict,idf_dict,stopwords)

            # write doc
            docs_out.write(fullpath_doc_name+'\n')
            for i,sent in enumerate(ref_sents):
                docs_out.write(' '.join([str(wid) for wid in sent]) +'\n')
                mx_sent_len = max(mx_sent_len,len(sent))
            docs_out.write('\n')

            mx_doc_len = max(mx_doc_len,len(ref_sents))

            # write question
            questions_out.write(fullpath_doc_name+'\n')
            questions_out.write(' '.join([str(wid) for wid in ref_question]) +'\n\n')
            
            # write labels
            labels_out.write(fullpath_doc_name+'\n')
            labels_out.write('\n'.join([str(lbl) for lbl in labels]) + '\n\n')

            # writing sentence cnt scores
            cnt_scores_out.write(fullpath_doc_name+'\n')
            cnt_scores_out.write('\n'.join(["%d" % x for x in cnt]) + '\n\n')

            # writing sentence isf scores
            isf_scores_out.write(fullpath_doc_name+'\n')
            isf_scores_out.write('\n'.join(["%.6f" % x for x in isf]) + '\n\n')

            # writing sentence idf scores
            idf_scores_out.write(fullpath_doc_name+'\n')
            idf_scores_out.write('\n'.join(["%.6f" % x for x in idf]) + '\n\n')

            # writing sentence local isf scores
            locisf_scores_out.write(fullpath_doc_name+'\n')
            locisf_scores_out.write('\n'.join(["%.6f" % x for x in locisf]) + '\n\n')

            ## INSERT QUERY EXPANSION HERE

            if count%10000 == 0:
                print "-->doc_count:",count
            count +=1
        print("%s: %d" %(corpus_split,count))
        print("Max document length (nsents) in %s set:%d" % (corpus_split,mx_doc_len))
        print("Max sentence length (nwords) in %s set:%d" % (corpus_split,mx_sent_len))
        print("# empty new labels: ",nempties)
