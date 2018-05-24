#import tensorflow as tf
#import numpy as np
#import time
#import argparse
#from tqdm import tqdm
#from utils.average_meter import FPSMeter
#import cv2
#from utils.img_utils import decode_labels
#import scipy.misc
#from tensorflow.core.framework import graph_pb2
#import copy
#
#def optimized_data_loader(h,w,sess):
#    with tf.device('/cpu:0'):
#        data_x = np.load("data/full_cityscapes_res/X_val.npy")
#
#        data_x_new = np.zeros((data_x.shape[0], h, w, 3),
#                                   dtype=np.uint8)
#        for i in range(data_x.shape[0]):
#            data_x_new[i] = scipy.misc.imresize(data_x[i], (h, w))
#
#        data_x = data_x_new
#        print(data_x.shape)
#        print(data_x.dtype)
#        print("DATA ITERATOR HERE!!")
#
#        features_placeholder = tf.placeholder(tf.float32, data_x.shape)
#
#        dataset = tf.contrib.data.Dataset.from_tensor_slices(features_placeholder)
#        dataset = dataset.batch(1)
#        iterator = tf.contrib.data.Iterator.from_structure(dataset.output_types,
#                                                        dataset.output_shapes)
#        next_batch = iterator.get_next()
#        training_init_op = iterator.make_initializer(dataset)
#        sess.run(training_init_op, feed_dict={features_placeholder: data_x})
#    return next_batch#, training_init_op
#
#parser = argparse.ArgumentParser(description="Inference test")
#parser.add_argument('--graph', type=str)
#parser.add_argument('--iterations', default=500, type=int)
#
## Parse the arguments
#args = parser.parse_args()
#
#if args.graph is not None:
#    with tf.gfile.GFile(args.graph, 'rb') as f:
#        graph_def = tf.GraphDef()
#        graph_def.ParseFromString(f.read())
#else:
#    raise ValueError("--graph should point to the input graph file.")
#
#G = tf.Graph()
#with tf.Session(graph=G) as sess:
#    x_in= optimized_data_loader(360, 640, sess)
#    # The inference is done till the argmax layer on the logits, as the softmax layer is not important.
##    y, = tf.import_graph_def(graph_def, input_map={'network/input/Placeholder':c}, return_elements=['network/output/ArgMax:0'])
#    y, = tf.import_graph_def(graph_def, input_map={'network/input/Placeholder':x_in}, return_elements=['network/output/ArgMax:0'])
#    #print('Operations in Graph:')
#    #print([op.name for op in G.get_operations()])
#    #x = G.get_tensor_by_name('import/network/input/Placeholder:0')
#
#    tf.global_variables_initializer().run()
#    #img = np.ones((1, 512, 1024, 3), dtype=np.uint8)
#    imgs= np.load('data/full_cityscapes_res/X_val.npy')
#    img= np.expand_dims(imgs[0,:,:,:], axis=0)
#    labels= np.load('data/full_cityscapes_res/Y_val.npy')
#    fps_meter = FPSMeter()
#    # Experiment should be repeated in order to get an accurate value for the inference time and FPS.
#    for _ in tqdm(range(args.iterations)):
#        start = time.time()
#        out = sess.run(y)#, feed_dict={x: img})
##        out2= decode_labels(out, 20)
##        label= decode_labels(np.expand_dims(labels[0,:,:],axis=0), 20)
##        cv2.imshow('img', img[0])
##        cv2.imshow('labels', label[0])
##        cv2.imshow('preds', out2[0])
##        cv2.waitKey()
#
#        fps_meter.update(time.time() - start)
#
#fps_meter.print_statistics()

import tensorflow as tf
import numpy as np
import time
import argparse
from tqdm import tqdm
from utils.average_meter import FPSMeter
import cv2
from utils.img_utils import decode_labels
import scipy.misc
from tensorflow.core.framework import graph_pb2
import copy

def optimized_data_loader(h,w,sess):
    with tf.device('/cpu:0'):
        data_x = np.load("data/full_cityscapes_res/X_val.npy")

        data_x_new = np.zeros((data_x.shape[0], h, w, 3),
                                   dtype=np.uint8)
        for i in range(data_x.shape[0]):
            data_x_new[i] = scipy.misc.imresize(data_x[i], (h, w))

        data_x = data_x_new
        print(data_x.shape)
        print(data_x.dtype)
        print("DATA ITERATOR HERE!!")

        features_placeholder = tf.placeholder(tf.float32, data_x.shape)

        dataset = tf.contrib.data.Dataset.from_tensor_slices(features_placeholder)
        dataset = dataset.batch(1)
        iterator = tf.contrib.data.Iterator.from_structure(dataset.output_types,
                                                        dataset.output_shapes)
        next_batch = iterator.get_next()
        training_init_op = iterator.make_initializer(dataset)
        sess.run(training_init_op, feed_dict={features_placeholder: data_x})
    return next_batch#, training_init_op

parser = argparse.ArgumentParser(description="Inference test")
parser.add_argument('--graph', type=str)
parser.add_argument('--iterations', default=500, type=int)

# Parse the arguments
args = parser.parse_args()

if args.graph is not None:
    with tf.gfile.GFile('graph_optimized.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
else:
    raise ValueError("--graph should point to the input graph file.")

G = tf.Graph()
with tf.Session(graph=G) as sess:
    x_in= optimized_data_loader(360,640, sess)
    # The inference is done till the argmax layer on the logits, as the softmax layer is not important.
    zeros= np.zeros((1,360,640,3))
    c = tf.constant(zeros, dtype=np.float32, shape=[1,360,640,3], name='network/input/Placeholder')
#    y, = tf.import_graph_def(graph_def, input_map={'network/input/Placeholder':c}, return_elements=['network/output/ArgMax:0'])
    y, = tf.import_graph_def(graph_def, input_map={'network/input/Placeholder':x_in}, return_elements=['network/output/ArgMax:0'])
    #print('Operations in Graph:')
    #print([op.name for op in G.get_operations()])
    #x = G.get_tensor_by_name('import/network/input/Placeholder:0')

    tf.global_variables_initializer().run()
    #img = np.ones((1, 512, 1024, 3), dtype=np.uint8)
    imgs= np.load('data/full_cityscapes_res/X_val.npy')
    img= np.expand_dims(imgs[0,:,:,:], axis=0)
    labels= np.load('data/full_cityscapes_res/Y_val.npy')
    fps_meter = FPSMeter()
    # Experiment should be repeated in order to get an accurate value for the inference time and FPS.
    for _ in tqdm(range(args.iterations)):
        start = time.time()
        out = sess.run(y)#, feed_dict={x: img})
#        out2= decode_labels(out, 20)
#        label= decode_labels(np.expand_dims(labels[0,:,:],axis=0), 20)
#        cv2.imshow('img', img[0])
#        cv2.imshow('labels', label[0])
#        cv2.imshow('preds', out2[0])
#        cv2.waitKey()

        fps_meter.update(time.time() - start)

fps_meter.print_statistics()

