####################################
# Author: Ronald Cardenas
# Date: July 2017
# Project: Document Modeling with External Attention for Sentence Extraction

####################################

'''
Extracts gold sentences, content, sentence labels, and questions from dataset
and writes them down to disk, one file per sample (id,content,question,answer)
'''

import os,sys
import pdb
import nltk
import io
from utils import *

if __name__ == "__main__":
    splits = ['training','validation']#,'test']
    anon_setting = False
    anon_pref = 'anonym' if anon_setting else 'org'

    for split in splits:
        gold_final_dir = 'gold-squad-%s-%s' %(split,anon_pref)
        gold_final_dir = os.path.join(GOLD_SUMM_DIR,gold_final_dir)
        if not os.path.exists(gold_final_dir):
            os.makedirs(gold_final_dir)
        data_gen = read_data(split)
        count = 0
        for sample in data_gen:
            sample_id,sents,question,labels = sample.unpack()
            nsents = len(sents)

            gold_file = io.open(os.path.join(gold_final_dir,sample_id+'.gold'),'w',encoding='utf-8')
            gold_file.write('\n'.join([' '.join(sents[i]) for i in xrange(nsents) if labels[i]==1 ]))
            gold_file.close()

            doc_file = io.open(os.path.join(BASE_DIR,split,'%s.summary.final' % (sample_id)),'w',encoding='utf-8')
            doc_file.write( '\n'.join([' '.join(sent) for sent in sents]))
            doc_file.close()

            labels_file = open(os.path.join(BASE_DIR,split,'%s.summary.label' % (sample_id)),'w')
            labels_file.write('\n'.join([str(lbl) for lbl in labels]))
            labels_file.close()

            question_file = io.open(os.path.join(BASE_DIR,split,'%s.summary.question' % (sample_id)),'w',encoding='utf-8')
            question_file.write(' '.join(question))
            question_file.close()

            ### insert query expansion side info HERE
            if count%10000 == 0:
                print " ->",count
            count += 1
