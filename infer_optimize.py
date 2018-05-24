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
    print([op.name for op in G.get_operations()])
    x = G.get_tensor_by_name('import/network/input/Placeholder:0')

    tf.global_variables_initializer().run()

    img = np.ones((1, 360, 640, 3), dtype=np.uint8)
    fps_meter = FPSMeter()
    # Experiment should be repeated in order to get an accurate value for the inference time and FPS.
    for _ in tqdm(range(args.iterations)):
        start = time.time()
        out = sess.run(y, feed_dict={x: img})
        fps_meter.update(time.time() - start)

fps_meter.print_statistics()

