
# coding: utf-8

# # SAMPLE CHAR-RNN 
# ### LINUX KERNEL SOURCE

# In[1]:

import os
import time
from six.moves import cPickle
import numpy as np
import tensorflow as tf
print ("PACKAGES LOADED")


# # LOAD

# In[2]:

load_dir    = "data/linux_kernel"
load_name = os.path.join(load_dir, 'chars_vocab.pkl')
with open(load_name, 'rb') as fload:
    chars, vocab = cPickle.load(fload)
    print ("CHARS AND VOCAB ARE LOADED FROM [%s]" % (load_name))
load_name = os.path.join(load_dir, 'corpus_data.pkl')
with open(load_name, 'rb') as fload:
    corpus, data = cPickle.load(fload)
    print ("CORPUS AND DATA ARE LOADED FROM [%s]" % (load_name))


# # BUILD SEQ2SEQ MODEL

# In[3]:

batch_size = 1
seq_length = 1
vocab_size = len(vocab)
rnn_size   = 128
num_layers = 2
grad_clip  = 5. 

# CONSTRUCT RNN MODEL
unitcell   = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
cell       = tf.nn.rnn_cell.MultiRNNCell([unitcell] * num_layers)
input_data = tf.placeholder(tf.int32, [batch_size, seq_length])
targets    = tf.placeholder(tf.int32, [batch_size, seq_length])
istate     = cell.zero_state(batch_size, tf.float32)

# WEIGHT
with tf.variable_scope('rnnlm') as scope:
    # SOFTMAX
    try:
        softmax_w = tf.get_variable("softmax_w", [rnn_size, vocab_size])
        softmax_b = tf.get_variable("softmax_b", [vocab_size])
    except ValueError:
        scope.reuse_variables()
        softmax_w = tf.get_variable("softmax_w", [rnn_size, vocab_size])
        softmax_b = tf.get_variable("softmax_b", [vocab_size])
    # EMBEDDING MATRIX
    embedding = tf.get_variable("embedding", [vocab_size, rnn_size])
    # tf.split(split_dim, num_split, value, name='split')
    inputs = tf.split(1, seq_length, tf.nn.embedding_lookup(embedding, input_data))
    # tf.squeeze(input, axis=None, name=None, squeeze_dims=None)
    inputs = [tf.squeeze(_input, [1]) for _input in inputs]

# DECODER
outputs, last_state = tf.nn.seq2seq.rnn_decoder(inputs, istate, cell
                , loop_function=None, scope='rnnlm')
output = tf.reshape(tf.concat(1, outputs), [-1, rnn_size])
logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
probs  = tf.nn.softmax(logits)

# LOSS
loss = tf.nn.seq2seq.sequence_loss_by_example([logits], # Input
    [tf.reshape(targets, [-1])], # Target
    [tf.ones([batch_size * seq_length])], # Weight 
    vocab_size)

# OPTIMIZER
cost     = tf.reduce_sum(loss) / batch_size / seq_length
final_state = last_state
lr       = tf.Variable(0.0, trainable=False)
tvars    = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), grad_clip)
_optm    = tf.train.AdamOptimizer(lr)
optm     = _optm.apply_gradients(zip(grads, tvars))

print ("NETWORK READY")


# # RESTORE MODEL

# In[4]:

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(tf.global_variables())
ckpt  = tf.train.get_checkpoint_state(load_dir)

print (ckpt.model_checkpoint_path)
saver.restore(sess, ckpt.model_checkpoint_path)


# # GENERATE

# In[8]:

# Sampling function
def weighted_pick(weights):
    t = np.cumsum(weights)
    s = np.sum(weights)
    return(int(np.searchsorted(t, np.random.rand(1)*s)))

# Sample using RNN and prime characters
prime = "/* "
state = sess.run(cell.zero_state(1, tf.float32))
for char in prime[:-1]:
    x = np.zeros((1, 1))
    x[0, 0] = vocab[char]
    state = sess.run(final_state, feed_dict={input_data: x, istate:state})

# Sample 'num' characters
ret  = prime
char = prime[-1] # <= This goes IN! 
num  = 2000
for n in range(num):
    x = np.zeros((1, 1))
    x[0, 0] = vocab[char]
    [probsval, state] = sess.run([probs, final_state]
        , feed_dict={input_data: x, istate:state})
    p      = probsval[0] 
    sample = weighted_pick(p)
    # sample = np.argmax(p)
    pred   = chars[sample]
    ret    = ret + pred
    char   = pred
    
print ("_"*100)
print (ret)
print ("_"*100)

