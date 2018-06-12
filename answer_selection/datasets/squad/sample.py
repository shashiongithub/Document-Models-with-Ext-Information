import nltk

class Sample(object):
  def __init__(self,sid,content,question,labels):
    self.sid = sid
    sents = nltk.sent_tokenize(content)
    self.content = []
    for sent in sents:
      self.content.append( nltk.word_tokenize(sent) )
    self.question = nltk.word_tokenize(question)
    self.labels = labels

  def unpack(self):
    return self.sid,\
           self.content, \
           self.question, \
           self.labels
