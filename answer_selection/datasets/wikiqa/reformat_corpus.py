####################################
# Author: Ronald Cardenas
# Date: July 2017
# Project: Document Modeling with External Attention for Sentence Extraction

####################################

'''
WikiQA dataset
Use author's code to generate wiki_cnn.pkl object with the whole dataset
Output format suitable for SideNET
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
    vocab = get_modified_vocab(force=force)
    #vocab = get_local_vocab(force=force)
    #stopwords = get_stopwords_ids(vocab)
    stopword_ids = get_short_stpw_ids(vocab)
    stopwords = get_short_stpw()

    isf_dict,idf_dict = get_isf_idf_dict(vocab,force=force)
    #idfbase_dict = get_idf_baseline_dict(stopwords,force=force)

    for corpus_split in splits:
        mx_doc_len = 0
        mx_sent_len = 0
        #sentById,questionsById,labelsById,featsById = read_data(corpus_split,True)
        sentById,questionsById,labelsById,featsById = read_from_txt(corpus_split,True,get_not_ans=False)

        # outputs files
        docs_out      = open(os.path.join(PREPROC_DATA_DIR,'wikiqa',"%s.org_ent.doc"      % (corpus_split)), 'w')
        questions_out = open(os.path.join(PREPROC_DATA_DIR,'wikiqa',"%s.org_ent.question" % (corpus_split)), 'w')
        labels_out    = open(os.path.join(PREPROC_DATA_DIR,'wikiqa',"%s.org_ent.label"    % (corpus_split)), 'w')
        isf_scores_out    = open(os.path.join(PREPROC_DATA_DIR,'wikiqa',"%s.isf.scores" % (corpus_split)), 'w')
        locisf_scores_out = open(os.path.join(PREPROC_DATA_DIR,'wikiqa',"%s.locisf.scores" % (corpus_split)), 'w')
        idf_scores_out    = open(os.path.join(PREPROC_DATA_DIR,'wikiqa',"%s.idf.scores" % (corpus_split)), 'w')
        cnt_scores_out    = open(os.path.join(PREPROC_DATA_DIR,'wikiqa',"%s.cnt.scores" % (corpus_split)), 'w')
        wgt_scores_out    = open(os.path.join(PREPROC_DATA_DIR,'wikiqa',"%s.wgtcnt.scores" % (corpus_split)), 'w')
        
        # write to output files
        count = 0
        for _id,question in questionsById.items():
            sents = sentById[_id]
            labels = labelsById[_id]
            
            sample_id = get_hash(question)
            fullpath_doc_name = os.path.join(BASE_DIR,corpus_split,sample_id+'.summary.final')

            ref_sents = words_to_id(sents,vocab)
            ref_question = words_to_id([question],vocab)[0]
            
            isf,idf,locisf,hand_cnts = eval_cnts(ref_question,ref_sents,isf_dict,idf_dict,stopword_ids)
            #cnt,wgt = eval_cnts_base(question,sents,idfbase_dict,stopwords)
            if len(featsById[_id])>0:
                feats = featsById[_id]
                cnt = [x[0] for x in feats]
                wgt = [x[1] for x in feats]
            else:
                cnt,wgt = hand_cnts,idf

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

            # writing sentence wgt scores
            wgt_scores_out.write(fullpath_doc_name+'\n')
            wgt_scores_out.write('\n'.join(["%.6f" % x for x in wgt]) + '\n\n')

            # writing sentence isf scores
            isf_scores_out.write(fullpath_doc_name+'\n')
            isf_scores_out.write('\n'.join(["%.6f" % x for x in isf]) + '\n\n')

            # writing sentence idf scores
            idf_scores_out.write(fullpath_doc_name+'\n')
            idf_scores_out.write('\n'.join(["%.6f" % x for x in idf]) + '\n\n')

            # writing sentence local isf scores
            locisf_scores_out.write(fullpath_doc_name+'\n')
            locisf_scores_out.write('\n'.join(["%.6f" % x for x in locisf]) + '\n\n')

            if count%500 == 0:
                print "-->doc_count:",count
            count +=1
        print("Max document length (nsents) in %s set:%d" % (corpus_split,mx_doc_len))
        print("Max sentence length (nwords) in %s set:%d" % (corpus_split,mx_sent_len))
        print("Num samples: %d" % count)
