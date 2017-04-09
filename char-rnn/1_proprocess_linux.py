
# coding: utf-8

# # PREPROCESS TEXT

# In[49]:

import os
import collections
from six.moves import cPickle
import numpy as np
print ("PACKAGES LOADED")


# # LOAD TEXT

# In[2]:

data_dir    = "data/linux_kernel"
save_dir    = "data/linux_kernel"
input_file  = os.path.join(data_dir, "input.txt")
with open(input_file, "r") as f:
    data = f.read()
print ("TYPE OF DATA IS %s" % (type(data)))
print ("TEXT LOADED FROM [%s]" % (input_file))


# # COUNT CHARACTERS

# In[22]:

counter = collections.Counter(data)
print ("TYPE OF 'COUNTER.ITEM()' IS [%s] AND LENGTH IS [%d]" 
       % (type(counter.items()), len(counter.items()))) 
for i in range(5):
    print ("[%d]TH ELEMENT IS [%s]" % (i, counter.items()[i]))


# # SORT CHARACTER COUNTS

# In[23]:

count_pairs = sorted(counter.items(), key=lambda x: -x[1]) 
print ("TYPE OF 'COUNT_PAIRS' IS [%s] AND LENGTH IS [%d]" 
       % (type(count_pairs), len(count_pairs))) 
for i in range(5):
    print ("[%d]TH ELEMENT IS [%s]" % (i, count_pairs[i]))


# # MAKE DICTIONARY
# ## : CHARS & VOCAB

# In[24]:

chars, counts = zip(*count_pairs)
vocab = dict(zip(chars, range(len(chars))))
print ("TYPE OF 'CHARS' IS [%s] AND LENGTH IS [%d]" 
    % (type(chars), len(chars))) 
print ("TYPE OF 'COUNTS' IS [%s] AND LENGTH IS [%d]" 
    % (type(counts), len(counts))) 
print ("TYPE OF 'VOCAB' IS [%s] AND LENGTH IS [%d]" 
    % (type(vocab), len(vocab))) 


# # USAGE OF 'CHARS' AND 'VOCAB

# In[30]:

# CHARS: NUMBER -> CHAR
print ("==========CHARS USAGE==========")
for i in range(5):
    print (" [%d/%d]" % (i, 3)), # COMMA STOPS LINE CHANGE
    print ("CHARS[%d] IS [%s]" % (i, chars[i]))
# VOCAB: CHAR -> NUMBER
print ("==========VOCAB USAGE==========")
for i in range(5):
    print (" [%d/%d]" % (i, 3)), # <= This comma remove '\n'
    print ("VOCAB[%s] IS [%s]" % (chars[i], vocab[chars[i]]))


# # SAVE CHARS AND VOCAB

# In[44]:

save_name = os.path.join(save_dir, 'chars_vocab.pkl')
with open(save_name, 'wb') as fsave:
    cPickle.dump((chars, vocab), fsave)
    print ("CHARS AND VOCAB ARE SAVED TO [%s]" % (save_name))

# LOAD 
load_name = os.path.join(save_dir, 'chars_vocab.pkl')
with open(load_name, 'rb') as fload:
    chars2, vocab2 = cPickle.load(fload)
    print ("CHARS AND VOCAB ARE LOADED FROM [%s]" % (load_name))
# CHARS: NUMBER -> CHAR
print ("==========CHARS2==========")
for i in range(5):
    print (" [%d/%d]" % (i, 3)), # COMMA STOPS LINE CHANGE
    print ("CHARS2[%d] IS [%s]" % (i, chars2[i]))
# VOCAB: CHAR -> NUMBER
print ("==========VOCAB2==========")
for i in range(5):
    print (" [%d/%d]" % (i, 3)), # <= This comma remove '\n'
    print ("VOCAB2[%s] IS [%s]" % (chars2[i], vocab2[chars2[i]]))


# # DATA => CORPUS

# In[60]:

corpus = np.array(list(map(vocab.get, data)))
print ("TYPE OF 'DATA' IS [%s] AND LENGTH IS [%d]" %(type(data), len(data)))
print ("TYPE OF 'CORPUS' IS [%s] AND LENGTH IS [%d]" %(type(corpus), len(data)))

print ("============DATA LOOKS LIKE============")
print (data[:50])
print ("============CORPUS LOOKS LIKE============")
print (corpus[:50])


# # SAVE

# In[61]:

save_name = os.path.join(save_dir, 'corpus_data.pkl')
with open(save_name, 'wb') as fsave:
    cPickle.dump((corpus, data), fsave)
    print ("CORPUS AND DATA ARE SAVED TO [%s]" % (save_name)) 

