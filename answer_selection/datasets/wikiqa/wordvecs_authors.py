####################################
# Author: Ronald Cardenas
# Date: July 2017
# Project: Document Modeling with External Attention for Sentence Extraction

####################################

'''
WikiQA (Yang et al) snippet of original authors code
necessary to upload data pickle file generated by the author's preprocessor
'''

import gensim
from gensim.models.doc2vec import LabeledSentence, Doc2Vec


class WordVecs(object):
    """
    precompute embeddings for word/feature/tweet etc.
    """
    def __init__(self, fname, vocab, binary=1, has_header=False):
        if binary == 1:
            word_vecs = self.load_bin_vec(fname, vocab)
        else:
            word_vecs = self.load_txt_vec(fname, vocab, has_header)
        self.k = len(word_vecs.values()[0])
        self.add_unknown_words(word_vecs, vocab, k=self.k)
        self.W, self.word_idx_map = self.get_W(word_vecs, k=self.k)

    def get_W(self, word_vecs, k=300):
        """
        Get word matrix. W[i] is the vector for word indexed by i
        """
        vocab_size = len(word_vecs)
        word_idx_map = dict()
        W = np.zeros(shape=(vocab_size+1, k))            
        W[0] = np.zeros(k)
        i = 1
        for word in word_vecs:
            W[i] = word_vecs[word]
            word_idx_map[word] = i
            i += 1
        return W, word_idx_map

    def load_bin_vec(self, fname, vocab):
        """
        Loads 300x1 word vecs from Google (Mikolov) word2vec
        """
        word_vecs = {}
        with open(fname, "rb") as f:
            header = f.readline()
            vocab_size, layer1_size = map(int, header.split())
            binary_len = np.dtype('float32').itemsize * layer1_size
            for line in xrange(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)   
                if word in vocab:
                   word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
                else:
                    f.read(binary_len)
        logger.info("num words already in word2vec: " + str(len(word_vecs)))
        return word_vecs
    
    def load_txt_vec(self, fname, vocab, has_header=False):
        """
        Loads 50x1 word vecs from sentiment word embeddings (Tang et al., 2014)
        """
        word_vecs = {}
        pos = 0
        with open(fname, "rb") as f:
            if has_header: header = f.readline()
            for line in f:
                parts = line.strip().split()
                word = parts[0]
                if word in vocab:
                   word_vecs[word] = np.asarray(map(float, parts[1:]))
                pos += 1
        logger.info("num words already in word2vec: " + str(len(word_vecs)))
        return word_vecs

    def add_unknown_words(self, word_vecs, vocab, min_df=1, k=300):
        """
        For words that occur in at least min_df documents, create a separate word vector.    
        0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
        """
        for word in vocab:
            if word not in word_vecs and vocab[word] >= min_df:
                #print word
                word_vecs[word] = np.random.uniform(-0.25,0.25,k)  