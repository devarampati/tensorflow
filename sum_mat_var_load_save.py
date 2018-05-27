#multiply and accumulate matrices
#NOTE : Mention the names to variables. Otherwise TF can load wrong values if shape matches
import tensorflow as tf
import numpy as np
ckpt_file = "./x.ckpt"
graph =tf.Graph()
x_s = np.array([[2,2],[2,2]],np.float32)
y_s = [2,2]
with graph.as_default() as g:
  x = tf.Variable(x_s,name="x_var")
  y = tf.placeholder(tf.float32,y_s)
  z = tf.matmul(x,y)
  w = tf.reduce_sum(z)

saver = tf.train.Saver([x])
y_ = np.array([[3,3],[3,3]])
update_x = tf.assign(x,x+1)

with tf.Session(graph=g) as sess:
  sess.run(tf.global_variables_initializer())
  mat_,sum_ = sess.run([z,w],feed_dict={y:y_})
  sess.run([update_x])
  print("Mul: ", mat_)
  print("Sum: ", sum_)
  saver.save(sess, save_path=ckpt_file)

with tf.Session(graph=g) as sess:
  saver.restore(sess,save_path=ckpt_file)
  for i in range(4):
    mat_,sum_ = sess.run([z,w],feed_dict={y:y_})
    sess.run([update_x])
    print("Mul: ", mat_)
    print("Sum: ", sum_)
  saver.save(sess, save_path=ckpt_file)
