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

    def __init__(self, args, sess, model):
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
        ##################################################################################
        # Init summaries

        # Summary variables
        self.scalar_summary_tags = []
        self.images_summary_tags = [('prediction_sample', [None, self.params.img_height, self.params.img_width * 3, 3])]

        self.summary_tags = self.scalar_summary_tags + self.images_summary_tags
        self.summary_placeholders = {}
        self.summary_ops = {}
        # init summaries and it's operators
        self.init_summaries()
        # Create summary writer
        self.summary_writer = tf.summary.FileWriter(self.args.summary_dir, self.sess.graph)
        ##################################################################################
        # Init load data and generator
        self.generator = None
        self.num_iterations_per_epoch = None  # It will be calculated in loading the data
        if self.args.data_mode == "experiment":
            print("ERROR this data_mode is not implemented..")
            exit(-1)
        elif self.args.data_mode == "overfit":
            self.train_data = None
            self.train_data_len = None
            self.load_overfit_data()
            self.generator = self.overfit_generator
        else:
            print("ERROR Please select a proper data_mode BYE")
            exit(-1)

            ##################################################################################

    def load_overfit_data(self):
        print("Loading data..")
        self.train_data = {'X': np.load(self.args.data_dir + "X.npy"),
                           'Y': np.load(self.args.data_dir + "Y.npy")}
        self.train_data_len = self.train_data['X'].shape[0]
        self.num_iterations_per_epoch = (self.train_data_len + self.args.batch_size - 1) // self.args.batch_size
        print("Train-shape-x -- " + str(self.train_data['X'].shape))
        print("Train-shape-y -- " + str(self.train_data['Y'].shape))
        print("Num of iterations in one epoch -- " + str(self.num_iterations_per_epoch))
        print("Overfitting data is loaded")

    def overfit_generator(self):
        start = 0
        new_epoch_flag = True
        idx = None
        while True:
            # init index array if it is a new_epoch
            if new_epoch_flag:
                if self.args.shuffle:
                    idx = np.random.choice(self.train_data_len, self.train_data_len, replace=False)
                else:
                    idx = np.arange(self.train_data_len)
                new_epoch_flag = False

            # select the mini_batches
            mask = idx[start:start + self.args.batch_size]
            x_batch = self.train_data['X'][mask]
            y_batch = self.train_data['Y'][mask]

            start += self.args.batch_size
            if start >= self.train_data_len:
                start = 0
                new_epoch_flag = True

            yield x_batch, y_batch

    def init_summaries(self):
        """
        Create the summary part of the graph
        :return:
        """
        with tf.variable_scope('train-summary'):
            for tag in self.scalar_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag)
                self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])
            for tag, shape in self.images_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', shape, name=tag)
                self.summary_ops[tag] = tf.summary.image(tag, self.summary_placeholders[tag], max_outputs=10)

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
        if summaries_merged is not None:
            self.summary_writer.add_summary(summaries_merged, step)

    def load_train_data(self):
        pass

    def train_generator(self):
        pass

    def train(self):
        pass

    def test_per_epoch(self, step, epoch):
        pass

    def overfit(self):
        print("Overfitting mode will begin NOW..")

        for cur_epoch in range(self.model.global_epoch_tensor.eval(self.sess) + 1, self.args.num_epochs + 1, 1):

            # init tqdm and get the epoch value
            tt = tqdm(self.generator(), total=self.num_iterations_per_epoch, desc="epoch-" + str(cur_epoch) + "-")

            # init the current iterations
            cur_iteration = 0

            # loop by the number of iterations
            for x_batch, y_batch in tt:

                # get the cur_it for the summary
                cur_it = self.model.global_step_tensor.eval(self.sess)

                # Feed this variables to the network
                feed_dict = {self.model.x_pl: x_batch,
                             self.model.y_pl: y_batch,
                             self.model.is_training: True
                             }

                # Run the feed forward but the last iteration finalize what you want to do
                if cur_iteration < self.num_iterations_per_epoch - 1:

                    # run the feed_forward
                    _, summaries_merged = self.sess.run(
                        [self.model.train_op, self.model.merged_summaries],
                        feed_dict=feed_dict)

                    # summarize
                    self.add_summary(cur_it, summaries_merged=summaries_merged)

                else:

                    # run the feed_forward
                    _, loss, acc, summaries_merged, segmented_imgs = self.sess.run(
                        [self.model.train_op, self.model.loss, self.model.accuracy,
                         self.model.merged_summaries, self.model.segmented_summary],
                        feed_dict=feed_dict)
                    # summarize
                    summaries_dict = dict()
                    summaries_dict['prediction_sample'] = segmented_imgs
                    self.add_summary(cur_it, summaries_dict=summaries_dict, summaries_merged=summaries_merged)

                    # Update the Global step
                    self.model.global_step_assign_op.eval(session=self.sess,
                                                          feed_dict={self.model.global_step_input: cur_it + 1})

                    # Update the Cur Epoch tensor
                    # it is the last thing because if it is interrupted it repeat this
                    self.model.global_epoch_assign_op.eval(session=self.sess,
                                                           feed_dict={self.model.global_epoch_input: cur_epoch + 1})

                    # print in console
                    tt.close()
                    print("epoch-" + str(cur_epoch) + "-" + "loss:" + str(loss) + "-" + "acc-" + str(acc)[:6])

                    # Break the loop to finalize this epoch
                    break

                # Update the Global step
                self.model.global_step_assign_op.eval(session=self.sess,
                                                      feed_dict={self.model.global_step_input: cur_it + 1})

                # update the cur_iteration
                cur_iteration += 1

            # Save the current checkpoint
            if cur_epoch != 0 and cur_epoch % self.args.save_every == 0:
                self.save_model()

        print("Overfitting Finished")

    def finalize(self):
        self.summary_writer.close()
        self.save_model()
