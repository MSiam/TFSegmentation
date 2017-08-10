"""
This class will take control of the whole process of training or testing Segmentation models
"""

import os
import time
import logging

import tensorflow as tf

from dirs import *
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

    def __init__(self):
        pass

    def run(self):
        print("Agent is running now...")
        # TODO Select the Mode

        # TODO Create Experiment Directory and the output directory and get hold of data dir

        # Reset the graph
        tf.reset_default_graph()

        # Create the session of the graph with some configuration
        # TODO Check that there is a running GPU
        # pass the gpu fraction as a paramter
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
        # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # TODO RUN the agent based on the type of the operation

        # sess.close()
        print("Agent is exited...")

    def train(self):
        pass

    def test(self):
        pass

    def create_dirs(self):
        pass
