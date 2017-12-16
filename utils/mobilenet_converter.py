import tensorflow as tf
from mobilenet_v1 import mobilenet_v1, mobilenet_v1_arg_scope
import pickle
import os

# This file should be put inside tensorflow/models/slim/nets or tensorflow/slim should be installed,
# it is used to dump the pretrained mobilenet model into .pkl weights file

slim = tf.contrib.slim
height = 224
width = 224
channels = 3
X = tf.placeholder(tf.float32, shape=[None, height, width, channels])
with slim.arg_scope(mobilenet_v1_arg_scope()):
    logits, end_points = mobilenet_v1(X, num_classes=1001, is_training=False)

sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, os.path.realpath(os.getcwd()) + "/mobilenet_v1_1.0_224.ckpt")

print("Saving the model to Pickle format: ")


def __save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


variables_to_train = tf.trainable_variables()
variables_executed = []
for variable in variables_to_train:
    variables_executed.append(sess.run(variable))
output_dict = {}
for i in range(len(variables_to_train)):
    output_dict[variables_to_train[i].name] = variables_executed[i]
__save_obj(output_dict, os.path.realpath(os.getcwd()) + "/variables.pkl")
print("Model Saved in variables.pkl")
