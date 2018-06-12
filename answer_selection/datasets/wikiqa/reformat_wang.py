####################################
# Author: Ronald Cardenas
# Date: July 2017
# Project: Document Modeling with External Attention for Sentence Extraction

####################################

"""
Samples in original Wiki txt files are not in the same order as pkl files
read pkl -> wang format (original WikiQA txt format)
"""

from __future__ import print_function

import os,sys
import pdb
import nltk
import io
from utils import *


if __name__ == "__main__":
    splits = ['training','validation','test']
    #splits = ['test']
    for corpus_split in splits:
        print("Reformating ",corpus_split,".....")
        #sentById,questionsById,labelsById = read_data(corpus_split,False)
        sentById,questionsById,labelsById = read_from_txt(corpus_split,False,get_not_ans=False)

        fn = "../../SeqMatchSeq_wang/data/wikiqa/"+corpus_split+".txt"
        #fn = "/home/ronald/SeqMatchSeq/data/wikiqa/WikiQACorpus/%s.txt" % corpus_split
        
        ## comprobado: no hay (q,cont) w diff labels en WikiQA

        output = open(fn,'w')
        counter = 1
        for _id,question in questionsById.items():
            sents = sentById[_id]
            labels = labelsById[_id]
            if sum(labels) == 0:
                #pdb.set_trace()
                continue
            for i,sent in enumerate(sents):
                output.write("%d\t%s\t%s\t%d\n" % (counter,question,sent,labels[i]))
                #output.write("%s\t%s\t%d\n" % (question,sent,labels[i]))
            counter += 1
        print("ninstances: ",counter-1)


