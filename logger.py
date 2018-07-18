# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514

import tensorflow as tf
import numpy as np
import scipy.misc
import os

try:
    from StringIO import StringIO
except ImportError:
    from io import BytesIO


class Logger(object):
    def __init__(self, log_dir):
        self.summary_writer = tf.summary.FileWriter(log_dir)
        self.summary_ops = {}

    def graph_summary(self, graph):
        self.summary_writer.add_graph(graph)
        self.summary_writer.flush()

    def scalar_summary(self, tag, value, step, scope):
        summary = tf.Summary(value=[tf.Summary.Value(tag=os.path.join(scope, tag), simple_value=value)])
        self.summary_writer.add_summary(summary, step)
        self.summary_writer.flush()

    def image_summary(self, tag, images, step, scope, max_output=4, random_summarization=False):
        """Log a list of images."""
        assert len(images.shape) == 4, "the input image shape should be in form [batch,hight,width,channels]"
        img_summaries = []
        if random_summarization:
            idxs = np.random.choice(images.shape[0], min(max_output, images.shape[0]))
            images = images[idxs]
        else:
            images = images[:max_output]
        if images.shape[-1]==1:
            images=np.squeeze(images)
        for i in range(images.shape[0]):
            img=images[i]
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag=os.path.join(scope, '%s/%d' % (tag, i)), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.summary_writer.add_summary(summary, step)
        self.summary_writer.flush()

    def histo_summary(self, tag, values, step, scope, bins=1000, ):
        """Log a histogram of the tensor of values."""
        counts, bin_edges = np.histogram(values, bins=bins)
        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=os.path.join(scope, tag), histo=hist)])
        self.summary_writer.add_summary(summary, step)
        self.summary_writer.flush()
        self.summary_writer.flush()

    # summarize tenorflow tenosrs or images or merged summary, but this requires tensorflow session run
    def summarize(self, sess, step, scope='train', summaries_dict=None, summaries_merged=None):
        """
        Add the summaries to tensorboard
        :param step:
        :param summaries_dict:
        :param summaries_merged:
        :return:
        """
        with tf.variable_scope(scope):
            if summaries_dict is not None:
                summary_list = []
                for tag, value in summaries_dict.items():
                    if tag not in self.summary_ops:
                        self.summary_placeholders[tag] = tf.placeholder('float32', value.shape, name=tag)
                        if len(value.shape) <= 1:
                            self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])
                        else:
                            self.summary_ops[tag] = tf.summary.image(tag, self.summary_placeholders[tag])

                    summary_list.append(sess.run(self.summary_ops[tag], {self.summary_placeholders[tag]: value}))

                for summary in summary_list:
                    self.summary_writer.add_summary(summary, step)
            if summaries_merged is not None:
                self.summary_writer.add_summary(summaries_merged, step)
            self.summary_writer.flush()