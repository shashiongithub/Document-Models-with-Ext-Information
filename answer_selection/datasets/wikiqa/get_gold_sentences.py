####################################
# Author: Ronald Cardenas
# Date: July 2017
# Project: Document Modeling with External Attention for Sentence Extraction

####################################

'''
WikiQA
Extracts gold sentences, content, sentence labels, and questions from dataset
and writes them down to disk, one file per sample (id,content,question,answer)
'''

import os,sys
import pdb
import nltk
import io
from utils import *

if __name__ == "__main__":
    splits = ['training','validation','test']
    anon_setting = False
    anon_pref = 'anonym' if anon_setting else 'org'

    for split in splits:
        gold_final_dir = 'gold-wikiqa-%s-%s' %(split,anon_pref)
        gold_final_dir = os.path.join(GOLD_SUMM_DIR,gold_final_dir)
        if not os.path.exists(gold_final_dir):
            os.makedirs(gold_final_dir)
        sentById,questionsById,labelsById = read_data(split)

        count = 0
        for _id,question in questionsById.items():
            sents = sentById[_id]
            labels = labelsById[_id]
            sample_id = get_hash(question)
            nsents = len(sents)

            if nsents == 0:
                pdb.set_trace()

            
            gold_file = open(os.path.join(gold_final_dir,sample_id+'.gold'),'w')
            gold_file.write('\n'.join([sents[i] for i in xrange(nsents) if labels[i]==1 ]))
            gold_file.close()

            doc_file = open(os.path.join(BASE_DIR,split,'%s.summary.final' % (sample_id)),'w')
            doc_file.write( '\n'.join(sents) )
            doc_file.close()

            labels_file = open(os.path.join(BASE_DIR,split,'%s.summary.label' % (sample_id)),'w')
            labels_file.write('\n'.join([str(lbl) for lbl in labels]))
            labels_file.close()

            question_file = open(os.path.join(BASE_DIR,split,'%s.summary.question' % (sample_id)),'w')
            question_file.write(question)
            question_file.close()

            ### insert query expansion side info HERE
            if count%10000 == 0:
                print " ->",count
            count += 1
