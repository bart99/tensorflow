
# coding: utf-8

# # TRAIN CHAR-RNN 
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


# # GENERATE XDATA AND YDATA

# In[3]:

batch_size  = 50
seq_length  = 200
num_batches = int(corpus.size / (batch_size * seq_length))
print ("NUM_BATCHES IS %s" % (num_batches))

corpus_reduced = corpus[:(num_batches*batch_size*seq_length)]
xdata = corpus_reduced
ydata = np.copy(xdata)
ydata[:-1] = xdata[1:]
ydata[-1]  = xdata[0]
print ('XDATA IS %s / TYPE IS %s / SHAPE IS %s' % (xdata, type(xdata), xdata.shape))
print ('YDATA IS %s / TYPE IS %s / SHAPE IS %s' % (ydata, type(ydata), ydata.shape))


# # GENERATE BATCH

# In[4]:

xbatches = np.split(xdata.reshape(batch_size, -1), num_batches, 1)
ybatches = np.split(ydata.reshape(batch_size, -1), num_batches, 1)

print ("BATCH_SIZE: %d / NUM_BATCHES: %d" % (batch_size, num_batches))
print ("TYPE OF 'XBATCHES' IS %s AND LENGTH IS %d" 
    % (type(xbatches), len(xbatches)))
print ("TYPE OF 'YBATCHES' IS %s AND LENGTH IS %d"
    % (type(ybatches), len(ybatches)))
print ("TYPE OF EACH BATCH IS %s AND SHAPE IS %s" 
    % (type(xbatches[0]), (xbatches[0]).shape))


# # XBATCHES & YBATCHES

# In[5]:

print ("===========XBATCHES===========")
print (xbatches[0])
print ("===========YBATCHES===========")
print (ybatches[0])


# # BUILD SEQ2SEQ MODEL

# In[6]:

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


# # TRAIN THE MODEL

# In[11]:

save_dir      = "data/linux_kernel"
num_epochs    = 200
print_every   = 500
save_every    = 1000
learning_rate = 0.001
decay_rate    = 0.97

sess = tf.Session()
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter(save_dir, graph=sess.graph)
saver = tf.train.Saver(tf.global_variables())
init_time = time.time()
for epoch in range(num_epochs): # FOR EACH EPOCH
    sess.run(tf.assign(lr, learning_rate * (decay_rate ** epoch)))
    state     = sess.run(istate)
    randbatchidx = np.random.permutation(num_batches)
    for iteration in range(num_batches): # FOR EACH ITERATION
        xbatch       = xbatches[randbatchidx[iteration]]
        ybatch       = ybatches[randbatchidx[iteration]]
        
        start_time   = time.time()
        train_loss, state, _ = sess.run([cost, final_state, optm]
            , feed_dict={input_data: xbatch, targets: ybatch, istate: state}) 
        total_iter = epoch*num_batches + iteration
        end_time   = time.time();
        duration   = end_time - start_time
        
        if total_iter % print_every == 0:
            print ("[%d/%d] cost: %.4f / Each batch learning took %.4f sec" 
                   % (total_iter, num_epochs*num_batches, train_loss, duration))
        if total_iter % save_every == 0: 
            ckpt_path = os.path.join(save_dir, 'model.ckpt')
            saver.save(sess, ckpt_path, global_step = total_iter)
            print("model saved to '%s'" % (ckpt_path)) 

