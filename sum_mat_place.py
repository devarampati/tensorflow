#multiply and accumulate matrices
import tensorflow as tf
import numpy as np

graph =tf.Graph()
x_s = [2,2]
y_s = [2,2]
with graph.as_default() as g:
  x = tf.placeholder(tf.float32,x_s)
  y = tf.placeholder(tf.float32,y_s)
  z = tf.matmul(x,y)
  w = tf.reduce_sum(z)
x_ = np.array([[2,2],[2,2]])
y_ = np.array([[3,3],[3,3]])
with tf.Session(graph=g) as sess:
  mat_,sum_ = sess.run([z,w],feed_dict={x:x_,y:y_})
  
print("Mul: ", mat_)
print("Sum: ", sum_)


