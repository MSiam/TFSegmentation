"""
This class will take control of the whole process of training or testing Segmentation models
"""

import tensorflow as tf

from models import *
from train import *
from test import *


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

    def run(self):
        """
        Initiate the Graph, sess, model, operator
        :return:
        """
        print("Agent is running now...")

        # Reset the graph
        tf.reset_default_graph()

        # Create the sess
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True,
                                                     gpu_options=gpu_options))

        # Create Model class
        self.model = self.model()
        self.operator = self.operator()

        if self.mode == 'train_n_test':
            self.train()
            self.test()
        elif self.mode == 'train' or 'overfit':
            self.train()
        else:
            self.test()

        self.sess.close()
        print("Agent is exited...")

    def train(self, mode="normal"):
        pass

    def test(self):
        pass

    def overfit(self):
        pass

    def create_dirs(self):
        pass
