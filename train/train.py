"""
Trainer class to train Segmentation models
"""

from train.basic_train import BasicTrain

from tqdm import tqdm
import numpy as np
import tensorflow as tf


class Train(BasicTrain):
    """
    Trainer class
    """

    def __int__(self, args, sess, model):
        """
        Call the constructor of the base class
        init summaries
        init loading data
        :param args:
        :param sess:
        :param model:
        :return:
        """
        super().__init__(args, sess, model)

        # Summary variables
        self.scalar_summary_tags = []
        self.images_summary_tags = []
        # self.scalar_summary_tags.extend(['accuracy-per-epoch',
        #                                  'loss-per-epoch',])
        # self.scalar_summary_tags.extend(['test-images', 'test-acc', 'test-loss'])


        self.summary_tags = self.scalar_summary_tags + self.images_summary_tags
        self.summary_placeholders = {}
        self.summary_ops = {}
        # init summaries and it's operators
        self.init_summaries()
        # Create summary writer
        self.summary_writer = tf.summary.FileWriter(self.args.summary_dir, self.sess.graph)

    def add_summary(self, step, summaries_dict=None, summaries_merged=None):
        """
        Add the summaries to tensorboard
        :param step:
        :param summaries_dict:
        :param summaries_merged:
        :return:
        """
        if summaries_dict is not None:
            summary_list = self.sess.run([self.summary_ops[tag] for tag in summaries_dict.keys()], {self.summary_placeholders[tag]: value for tag, value in summaries_dict.items()})
            for summary in summary_list:
                self.summary_writer.add_summary(summary, step)
            self.summary_writer.flush()
        if summaries_merged is not None:
            self.summary_writer.add_summary(summaries_merged, step)
            self.summary_writer.flush()

    def init_summaries(self):
        """
        Create the summary part of the graph
        :return:
        """
        # TODO scalar summaries and image summaries
        with tf.variable_scope('train-summary'):
            for tag in self.scalar_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag)
                self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])

    def train(self):
        pass
