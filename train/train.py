"""
Trainer class to train Segmentation models
"""

from train.basic_train import BasicTrain
from metrics.metrics import Metrics

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
        self.scalar_summary_tags = ['mean_iou_on_val', 'mean_iou_on_test',
                                    'train-loss-per-epoch', 'val-loss-per-epoch',
                                    'train-acc-per-epoch', 'val-acc-per-epoch']
        self.images_summary_tags = [('train_prediction_sample', [None, self.params.img_height, self.params.img_width * 3, 3]),
                                    ('val_prediction_sample', [None, self.params.img_height, self.params.img_width * 3, 3])]

        self.summary_tags = []
        self.summary_placeholders = {}
        self.summary_ops = {}
        # init summaries and it's operators
        self.init_summaries()
        # Create summary writer
        self.summary_writer = tf.summary.FileWriter(self.args.summary_dir, self.sess.graph)
        ##################################################################################
        # Init load data and generator
        self.generator = None
        if self.args.data_mode == "experiment":
            self.train_data = None
            self.train_data_len = None
            self.val_data = None
            self.val_data_len = None
            self.num_iterations_training_per_epoch = None
            self.num_iterations_validation_per_epoch = None
            self.load_train_data()
            self.generator = self.train_generator
        elif self.args.data_mode == "test":
            self.test_data = None
            self.test_data_len = None
            self.load_test_data()
        elif self.args.data_mode == "overfit":
            self.train_data = None
            self.train_data_len = None
            self.num_iterations_per_epoch = None  # It will be calculated in loading the data
            self.load_overfit_data()
            self.generator = self.overfit_generator
        else:
            print("ERROR Please select a proper data_mode BYE")
            exit(-1)
        ##################################################################################
        # Init metrics class
        self.metrics = Metrics(self.args.num_classes)
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
        with tf.variable_scope('train-summary-per-epoch'):
            for tag in self.scalar_summary_tags:
                self.summary_tags += tag
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag)
                self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])
            for tag, shape in self.images_summary_tags:
                self.summary_tags += tag
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
        print("Loading Training data..")
        self.train_data = {'X': np.load(self.args.data_dir + "X_train.npy"),
                           'Y': np.load(self.args.data_dir + "Y_train.npy")}
        self.train_data_len = self.train_data['X'].shape[0]
        self.num_iterations_training_per_epoch = (self.train_data_len + self.args.batch_size - 1) // self.args.batch_size
        print("Train-shape-x -- " + str(self.train_data['X'].shape))
        print("Train-shape-y -- " + str(self.train_data['Y'].shape))
        print("Num of iterations on training data in one epoch -- " + str(self.num_iterations_training_per_epoch))
        print("Training data is loaded")

        print("Loading Validation data..")
        self.val_data = {'X': np.load(self.args.data_dir + "X_val.npy"),
                         'Y': np.load(self.args.data_dir + "Y_val.npy")}
        self.val_data_len = self.val_data['X'].shape[0]
        self.num_iterations_validation_per_epoch = (self.val_data_len + self.args.batch_size - 1) // self.args.batch_size
        print("Val-shape-x -- " + str(self.val_data['X'].shape))
        print("Val-shape-y -- " + str(self.val_data['Y'].shape))
        print("Num of iterations on validation data in one epoch -- " + str(self.num_iterations_validation_per_epoch))
        print("Validation data is loaded")

    def load_test_data(self):
        print("Loading Testing data..")
        self.test_data = {'X': np.load(self.args.data_dir + "X_test.npy"),
                          'Y': np.load(self.args.data_dir + "Y_test.npy")}
        self.test_data_len = self.test_data['X'].shape[0]
        print("Test-shape-x -- " + str(self.test_data['X'].shape))
        print("Test-shape-y -- " + str(self.test_data['Y'].shape))
        print("Test data is loaded")

    def train_generator(self):
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

            # update start idx
            start += self.args.batch_size

            if start >= self.train_data_len:
                start = 0
                new_epoch_flag = True

            yield x_batch, y_batch

    def train(self):
        print("Training mode will begin NOW ..")
        for cur_epoch in range(self.model.global_epoch_tensor.eval(self.sess) + 1, self.args.num_epochs + 1, 1):

            # init tqdm and get the epoch value
            tt = tqdm(self.generator(), total=self.num_iterations_training_per_epoch, desc="epoch-" + str(cur_epoch) + "-")

            # init the current iterations
            cur_iteration = 0

            # init acc and loss lists
            loss_list = []
            acc_list = []

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
                if cur_iteration < self.num_iterations_training_per_epoch - 1:

                    # run the feed_forward
                    _, loss, acc, summaries_merged = self.sess.run(
                        [self.model.train_op, self.model.loss, self.model.accuracy, self.model.merged_summaries],
                        feed_dict=feed_dict)
                    # log loss and acc
                    loss_list += loss
                    acc_list += acc
                    # summarize
                    self.add_summary(cur_it, summaries_merged=summaries_merged)

                else:

                    # run the feed_forward
                    _, loss, acc, summaries_merged, segmented_imgs = self.sess.run(
                        [self.model.train_op, self.model.loss, self.model.accuracy,
                         self.model.merged_summaries, self.model.segmented_summary],
                        feed_dict=feed_dict)
                    # log loss and acc
                    loss_list += loss
                    acc_list += acc
                    total_loss = np.mean(loss_list)
                    total_acc = np.mean(acc_list)
                    # summarize
                    summaries_dict = dict()
                    summaries_dict['train-loss-per-epoch'] = total_loss
                    summaries_dict['train-acc-per-epoch'] = total_acc
                    summaries_dict['train_prediction_sample'] = segmented_imgs
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
                    print("epoch-" + str(cur_epoch) + "-" + "loss:" + str(total_loss) + "-" + "acc-" + str(total_acc)[:6])

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

            # Test the model on validation
            if cur_epoch != 0 and cur_epoch % self.args.test_every == 0:
                self.test_per_epoch(step=self.model.global_step_tensor.eval(self.sess),
                                    epoch=self.model.global_epoch_tensor.eval(self.sess))

        print("Training Finished")

    def test_per_epoch(self, step, epoch):
        print("Validation at step:" + str(step) + " at epoch:" + str(epoch) + " ..")

        # init tqdm and get the epoch value
        tt = tqdm(range(self.num_iterations_validation_per_epoch), total=self.num_iterations_training_per_epoch, desc="Val-epoch-" + str(epoch) + "-")

        # init acc and loss lists
        loss_list = []
        acc_list = []

        # idx of minibatch
        idx = 0

        # reset metrics
        self.metrics.reset()

        # loop by the number of iterations
        for cur_iteration in tt:
            # load minibatches
            x_batch = self.val_data['X'][idx:idx + self.args.batch_size]
            y_batch = self.val_data['Y'][idx:idx + self.args.batch_size]

            # update idx of minibatch
            idx += self.args.batch_size

            # Feed this variables to the network
            feed_dict = {self.model.x_pl: x_batch,
                         self.model.y_pl: y_batch,
                         self.model.is_training: False
                         }

            # Run the feed forward but the last iteration finalize what you want to do
            if cur_iteration < self.num_iterations_validation_per_epoch - 1:

                # run the feed_forward
                out_argmax, loss, acc, summaries_merged = self.sess.run(
                    [self.model.out_argmax, self.model.loss, self.model.accuracy, self.model.merged_summaries],
                    feed_dict=feed_dict)
                # log loss and acc
                loss_list += loss
                acc_list += acc
                # log metrics
                self.metrics.update_metrics_batch(out_argmax, y_batch)

            else:

                # run the feed_forward
                out_argmax, loss, acc, summaries_merged, segmented_imgs = self.sess.run(
                    [self.model.out_argmax, self.model.loss, self.model.accuracy,
                     self.model.merged_summaries, self.model.segmented_summary],
                    feed_dict=feed_dict)
                # log loss and acc
                loss_list += loss
                acc_list += acc
                # log metrics
                self.metrics.update_metrics_batch(out_argmax, y_batch)
                # mean over batches
                total_loss = np.mean(loss_list)
                total_acc = np.mean(acc_list)
                mean_iou = self.metrics.compute_final_metrics(self.num_iterations_validation_per_epoch)
                # summarize
                summaries_dict = dict()
                summaries_dict['val-loss-per-epoch'] = total_loss
                summaries_dict['val-acc-per-epoch'] = total_acc
                summaries_dict['mean_iou_on_val'] = mean_iou
                summaries_dict['val_prediction_sample'] = segmented_imgs
                self.add_summary(step, summaries_dict=summaries_dict, summaries_merged=summaries_merged)

                # print in console
                tt.close()
                print("Val-epoch-" + str(epoch) + "-" + "loss:" + str(total_loss) + "-" + "acc-" + str(total_acc)[:6] + "-mean_iou-" + str(mean_iou))

                # Break the loop to finalize this epoch
                break

    def test(self):
        print("Testing mode will begin NOW..")

        # init tqdm and get the epoch value
        tt = tqdm(range(self.test_data_len))

        # init acc and loss lists
        loss_list = []
        acc_list = []
        img_list = []

        # idx of image
        idx = 0

        # reset metrics
        self.metrics.reset()

        # loop by the number of iterations
        for cur_iteration in tt:

            # load mini_batches
            x_batch = self.test_data['X'][idx:idx + 1]
            y_batch = self.test_data['Y'][idx:idx + 1]

            # update idx of mini_batch
            idx += 1

            # Feed this variables to the network
            feed_dict = {self.model.x_pl: x_batch,
                         self.model.y_pl: y_batch,
                         self.model.is_training: False
                         }

            # run the feed_forward
            out_argmax, loss, acc, summaries_merged, segmented_imgs = self.sess.run(
                [self.model.out_argmax, self.model.loss, self.model.accuracy,
                 self.model.merged_summaries, self.model.segmented_summary],
                feed_dict=feed_dict)

            # log loss and acc
            loss_list += loss
            acc_list += acc
            img_list += segmented_imgs[0]

            # log metrics
            self.metrics.update_metrics(out_argmax[0], y_batch[0], 0, 0)


        # mean over batches
        total_loss = np.mean(loss_list)
        total_acc = np.mean(acc_list)
        mean_iou = self.metrics.compute_final_metrics(self.test_data_len)

        # print in console
        tt.close()
        print("Here the statistics")
        print("Total_loss: " + str(total_loss))
        print("Total_acc: " + str(total_acc)[:6])
        print("mean_iou: " + str(mean_iou))

        # Break the loop to finalize this epoch
        # break

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
