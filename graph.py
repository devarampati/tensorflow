# This program helps to load and save computational graph
import tensorflow as tf
from google.protobuf import text_format

graph_file = "ab.pbtxt"
#graph_file = "abc.pb"
a = tf.constant(9)
b = tf.constant(4)
c = a*b+tf.constant(-4)
# To print all TF operations as a list
g1 = tf.get_default_graph()
#print(g1.get_operations())

# To load a saved pbtxt file 
g2def = tf.GraphDef()

if graph_file.endswith("pbtxt"):
    with open(graph_file) as f:
        text_format.Merge(f.read(), g2def)
else:
    with open(graph_file,"rb") as f:
        g2def.ParseFromString(f.read())

g2 = tf.Graph()
with g2.as_default() :
    tf.import_graph_def(g2def)

with tf.Session(graph=g2) as sess:
    add_c = g2.get_operation_by_name("import/add")
    print(sess.run(add_c.outputs[0]))

with tf.Session(graph=g1) as sess:
    print(sess.run(c))
    # To save the graphdef as a pbtxt file
    tf.train.write_graph(sess.graph_def, './', 'abc.pb',as_text=False)
