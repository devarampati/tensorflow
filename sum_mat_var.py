#multiply and accumulate matrices
import tensorflow as tf
import numpy as np

graph =tf.Graph()
x_s = np.array([[2,2],[2,2]],np.float32)
y_s = [2,2]
with graph.as_default() as g:
  x = tf.Variable(x_s)
  y = tf.placeholder(tf.float32,y_s)
  z = tf.matmul(x,y)
  w = tf.reduce_sum(z)

y_ = np.array([[3,3],[3,3]])
with tf.Session(graph=g) as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(4):
    mat_,sum_ = sess.run([z,w],feed_dict={y:y_})
    y_ = y_ + 1
    print("Mul: ", mat_)
    print("Sum: ", sum_)
    
  


