import nltk

class Sample(object):
  def __init__(self,sid,_sents,question,labels):
    self.sid = sid
    self.content = _sents
    self.question = nltk.word_tokenize(question)
    self.labels = labels

  def unpack(self):
    return self.sid,\
           self.content, \
           self.question, \
           self.labels
