from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import time
import random
#RNN output = act( input * W + state * U + B)
# input -> batch_zize X num_inputs
# State -> num_outputs X num_outputs
# W -> num_inputs X num_outputs
# U -> num_outputs X num_outputs
# B -> num_outputs

SEQ_LEN = 128
def generate_seq(val=64):
  n = val
  H = np.zeros([n, n])
  # Initialize Hadamard matrix of order n.
  i1 = 1
  while i1 < n:
    for i2 in range(i1):
      for i3 in range(i1):
        H[i2+i1][i3]    = H[i2][i3]
        H[i2][i3+i1]    = H[i2][i3]
        H[i2+i1][i3+i1] = not H[i2][i3]
    i1 += i1

  # Write the matrix.
  seq =  np.zeros([n,n])
  for i in range(n):
    for j in range(n):
      if H[i][j]:
         seq[i][j] = 1
      else:
         seq[i][j] = 0
    
  return(seq)

def generate_batch(step):
  label = np.zeros(SEQ_LEN)
  label[step%SEQ_LEN] = 1
  return (seq[step%SEQ_LEN],label)
batch_size = 1
input_size = SEQ_LEN
num_hidden = 64
num_classes = SEQ_LEN
num_training_steps = 102400
seq = generate_seq(SEQ_LEN)
graph = tf.Graph()
with graph.as_default() as g:
  x = tf.placeholder(tf.float32,[batch_size,input_size])
  y = tf.placeholder(tf.float32,[batch_size,num_classes])

  w = tf.Variable(tf.random_normal([num_hidden,num_classes]))
  b = tf.Variable(tf.random_normal([num_classes]))
  lstm_cell = rnn.BasicRNNCell(num_hidden)
  state = lstm_cell.zero_state(batch_size,tf.float32)
  output,state = lstm_cell(x,state)
  logits = tf.matmul(output,w) + b
  pred = tf.argmax(tf.nn.softmax(logits),1)
  
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y))
  opt = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
  train = opt.minimize(loss)
  init = tf.global_variables_initializer()


with tf.Session(graph=g) as sess:
  sess.run(init)
  accuracy = 0
  for step in range(num_training_steps):
    rand = random.randint(0,SEQ_LEN-1)
    in_data,label = generate_batch(rand)
    label = np.expand_dims(label,0)
    in_data = np.expand_dims(in_data,0)
    sess.run([train],feed_dict={x:in_data, y:label})
    current_loss,current_pred = sess.run([loss,pred],feed_dict = {x:in_data, y:label})
    print ("step,loss: ",step,current_loss)
    if current_pred == rand:
      accuracy = accuracy + 1
    if step%100 == 0:
      print ("accuracy: ",accuracy/100)
      time.sleep(0.5)
      accuracy = 0




