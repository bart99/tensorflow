
# coding: utf-8

# ## SIMPLE CHAR-RNN 

# In[1]:

from __future__ import print_function
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
tf.set_random_seed(0)  
print ("TENSORFLOW VERSION IS %s" % (tf.__version__))


# ## DEFINE TRAINING SEQUENCE

# In[2]:

sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")
print ("FOLLOWING IS OUR TRAINING SEQUENCE:")
print (sentence)


# ## DEFINE VOCABULARY AND DICTIONARY

# In[3]:

char_set = list(set(sentence))
char_dic = {w: i for i, w in enumerate(char_set)}
print ("VOCABULARY: ")
print (char_set)
print ("DICTIONARY: ")
print (char_dic)


# ## CONFIGURE NETWORK

# In[4]:

data_dim    = len(char_set)
num_classes = len(char_set)
hidden_size     = 64
sequence_length = 10  # Any arbitrary number


# ## SET TRAINING BATCHES

# In[5]:

dataX = []
dataY = []
for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i:i + sequence_length]
    y_str = sentence[i + 1: i + sequence_length + 1]
    x = [char_dic[c] for c in x_str]  # x str to index
    y = [char_dic[c] for c in y_str]  # y str to index
    dataX.append(x)
    dataY.append(y)
    print ("[%3d/%3d] [%s]=>[%s]" % (i, len(sentence), x_str, y_str))
    print ("%s%s=>%s" % (' '*10, x, y))


# In[6]:

ndata      = len(dataX)
batch_size = 20
print ("     'NDATA' IS %d" % (ndata))
print ("'BATCH_SIZE' IS %d" % (batch_size))


# ## DEFINE PLACEHOLDERS

# In[7]:

X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])
X_OH = tf.one_hot(X, num_classes)
print ("'sequence_length' IS [%d]" % (sequence_length))
print ("    'num_classes' IS [%d]" % (num_classes))
print("'X' LOOKS LIKE \n   [%s]" % (X))  
print("'X_OH' LOOKS LIKE \n   [%s]" % (X_OH))


# ## DEFINE MODEL

# In[8]:

with tf.variable_scope('CHAR-RNN', reuse=False):
    cell = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
    cell = rnn.MultiRNNCell([cell] * 2, state_is_tuple=True)
    # DYNAMIC RNN WITH FULLY CONNECTED LAYER
    _outputs, _states = tf.nn.dynamic_rnn(cell, X_OH, dtype=tf.float32)
    _outputs  = tf.contrib.layers.fully_connected(_outputs, num_classes, activation_fn=None)
    # RESHAPE FOR SEQUNCE LOSS
    outputs = tf.reshape(_outputs, [batch_size, sequence_length, num_classes])
print ("OUTPUTS LOOKS LIKE [%s]" % (outputs))
print ("MODEL DEFINED.")


# ## DEFINE TF FUNCTIONS

# In[9]:

# EQUAL WEIGHTS
weights = tf.ones([batch_size, sequence_length])
seq_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(seq_loss)
optm  = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
print ("FUNCTIONS DEFINED.")


# ## OPTIMIZE

# In[10]:

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(500):
    randidx = np.random.randint(low=0, high=ndata, size=batch_size)
    batchX = [dataX[iii] for iii in randidx]
    batchY = [dataY[iii] for iii in randidx]
    feeds = {X: batchX, Y: batchY}
    _, loss_val, results = sess.run(
        [optm, loss, outputs], feed_dict=feeds)
    if (i%100) == 0:
        for j, result in enumerate(results):
            index = np.argmax(result, axis=1)
            print(i, j, ''.join([char_set[t] for t in index]), loss_val)


# ### SAMPLING FUNCTION 

# In[17]:

LEN = 1;
# XL = tf.placeholder(tf.int32, [None, LEN])
XL     = tf.placeholder(tf.int32, [None, 1])
XL_OH  = tf.one_hot(XL, num_classes)
with tf.variable_scope('CHAR-RNN', reuse=True):
    cell_L = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
    cell_L = rnn.MultiRNNCell([cell_L] * 2, state_is_tuple=True)
    istate = cell_L.zero_state(batch_size=1, dtype=tf.float32)
    # DYNAMIC RNN WITH FULLY CONNECTED LAYER
    _outputs_L, states_L = tf.nn.dynamic_rnn(cell_L, XL_OH
                                , initial_state=istate, dtype=tf.float32)
    _outputs_L  = tf.contrib.layers.fully_connected(
        _outputs_L, num_classes, activation_fn=None)
    # RESHAPE FOR SEQUNCE LOSS
    outputs_L = tf.reshape(_outputs_L, [LEN, 1, num_classes])
print (XL)


# ## SAMPLE

# In[53]:

# BURNIN
prime = "if you "
istateval = sess.run(cell_L.zero_state(1, tf.float32))
for c in prime[:-1]:
    index = char_dic[c]
    inval = [[index]]
    outval, stateval = sess.run([outputs_L, states_L]
                        , feed_dict={XL:inval, istate:istateval})
    istateval = stateval


# In[58]:

# SAMPLE
inval  = [[char_dic[prime[-1]]]]
outval, stateval = sess.run([outputs_L, states_L]
                    , feed_dict={XL:inval, istate:istateval})
index = np.argmax(outval)
char  = char_set[index]
chars = ''
for i in range(500):
    inval = [[index]]
    outval, stateval = sess.run([outputs_L, states_L]
                        , feed_dict={XL:inval, istate:istateval})
    istateval = stateval
    index = np.argmax(outval)
    char  = char_set[index]
    chars += char

print ("SAMPLED SETENCE: \n %s" % (prime+chars))
print ("\nORIGINAL SENTENCE: \n %s" % (sentence))

