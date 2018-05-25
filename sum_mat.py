#multiply and accumulate matrices
import tensorflow as tf

graph =tf.Graph()
x_ = [3,3]
y_ = [2,2]
with graph.as_default() as g:
  x = tf.constant([x_,x_])
  y = tf.constant([y_,y_])
  z = tf.matmul(x,y)
  w = tf.reduce_sum(z)

with tf.Session(graph=g) as sess:
  mat_,sum_ = sess.run([z,w])
  
print("Mul: ", mat_)
print("Sum: ", sum_)


