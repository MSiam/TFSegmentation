"""
This class will take control of the whole process of training or testing Segmentation models
"""

import tensorflow as tf

from models import *
from train import *
from test import *
from utils.misc import timeit

import os
import pdb

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Agent:
    """
    Agent will run the program
    Choose the type of operation
    Create a model
    reset the graph and create a session
    Create a trainer or tester
    Then run it and handle it
    """

    def __init__(self, args):
        self.args = args
        self.mode = args.mode

        # Get the class from globals by selecting it by arguments
        self.model = globals()[args.model]
        self.operator = globals()[args.operator]

        self.sess = None

    @timeit
    def build_model(self):
        print('Building Train Network')
        with tf.variable_scope('network') as scope:
            self.train_model = self.model(self.args, phase=0)
            self.train_model.build()

        if self.mode == 'train':  # validation phase
            print('Building Test Network')
            with tf.variable_scope('network') as scope:
                scope.reuse_variables()
                self.test_model = self.model(self.args, phase=1)
                self.test_model.build()
        else:  # inference phase
            print('Building Test Network')
            with tf.variable_scope('network') as scope:
                scope.reuse_variables()
                self.test_model = self.model(self.args, phase=2)
                self.test_model.build()

    @timeit
    def run(self):
        """
        Initiate the Graph, sess, model, operator
        :return:
        """
        print("Agent is running now...\n\n")

        # Reset the graph
        tf.reset_default_graph()

        # Create the sess
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

        # Create Model class and build it
        with self.sess.as_default():
            self.build_model()
        # Create the operator
        self.operator = self.operator(self.args, self.sess, self.train_model, self.test_model)

        if self.mode == 'train_n_test':
            print("Sorry this mode is not available for NOW")
            exit(-1)
            # self.train()
            # self.test()
        elif self.mode == 'train':
            self.train()
        elif self.mode == 'overfit':
            self.overfit()
        elif self.mode == 'inference':
            self.inference()
        else:
            self.test()

        self.sess.close()
        print("\nAgent is exited...\n")

    def train(self):
        try:
            self.operator.train()
            self.operator.finalize()
        except KeyboardInterrupt:
            self.operator.finalize()

    def test(self):
        try:
            self.operator.test()
        except KeyboardInterrupt:
            pass

    def overfit(self):
        try:
            self.operator.overfit()
            self.operator.finalize()
        except KeyboardInterrupt:
            self.operator.finalize()

    def inference(self):
        try:
            self.operator.test_inference()
        except KeyboardInterrupt:
            pass
