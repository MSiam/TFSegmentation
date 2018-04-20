import tensorflow as tf
import numpy as np
import time
import argparse
from tqdm import tqdm
from utils.average_meter import FPSMeter

parser = argparse.ArgumentParser(description="Inference test")
parser.add_argument('--graph', type=str)
parser.add_argument('--iterations', default=1000, type=int)

# Parse the arguments
args = parser.parse_args()

if args.graph is not None:
    with tf.gfile.GFile(args.graph, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
else:
    raise ValueError("--graph should point to the input graph file.")

G = tf.Graph()

with tf.Session(graph=G) as sess:
    # The inference is done till the argmax layer on the logits, as the softmax layer is not important.
    y, = tf.import_graph_def(graph_def, return_elements=['network/output/ArgMax:0'])
    print('Operations in Graph:')
    for i, op in enumerate(G.get_operations()):
        print(op.name)
    print("-------------------------------------------")

    tf.global_variables_initializer().run()

    # Dataset API initialization (Doesn't work at the moment)
    # img = np.ones((args.iterations, 3, 360, 640), dtype=np.uint8)
    # x = G.get_tensor_by_name('import/DatasetAPI_features_placeholder:0')
    # init_op = G.get_operation_by_name('import/data_initializer')
    # sess.run(init_op, feed_dict={x: img})

    # Placeholder initialization
    x = G.get_tensor_by_name('import/network/input/Placeholder:0')
    is_training = G.get_tensor_by_name('import/network/input/Placeholder_2:0')
    img = np.ones((1, 3, 360, 640), dtype=np.uint8)

    fps_meter = FPSMeter()
    # Experiment should be repeated in order to get an accurate value for the inference time and FPS.
    for _ in tqdm(range(args.iterations)):
        start = time.time()
        out = sess.run(y, feed_dict={x: img, is_training: False})
        fps_meter.update(time.time() - start)

fps_meter.print_statistics()
