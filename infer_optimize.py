import tensorflow as tf
import numpy as np
import time

with tf.gfile.GFile('graph_optimized.pb', 'rb') as f:
   graph_def_optimized = tf.GraphDef()
   graph_def_optimized.ParseFromString(f.read())

G = tf.Graph()

with tf.Session(graph=G) as sess:
    y, = tf.import_graph_def(graph_def_optimized, return_elements=['network/output/Softmax:0'])
    print('Operations in Optimized Graph:')
    print([op.name for op in G.get_operations()])
    x = G.get_tensor_by_name('import/network/input/Placeholder:0')
#    import pdb; pdb.set_trace()
    tf.global_variables_initializer().run()
    img= np.ones((1,512,1024,3), dtype=np.uint8)
    start= time.time()
    out = sess.run(y, feed_dict={x: img})
    print('consumed time is ', time.time()-start)
#    print(out)

