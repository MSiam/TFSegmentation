"""
Trainer class to train Segmentation models
"""

from train.basic_train import BasicTrain
from metrics.metrics import Metrics
from utils.reporter import Reporter
from utils.misc import timeit

from tqdm import tqdm
import numpy as np
import tensorflow as tf
import matplotlib
import time
import h5py
from utils.augmentation import flip_randomly_left_right_image_with_annotation, \
    scale_randomly_image_with_annotation_with_fixed_size_output

matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
        self.scalar_summary_tags = ['mean_iou_on_val',
                                    'train-loss-per-epoch', 'val-loss-per-epoch',
                                    'train-acc-per-epoch', 'val-acc-per-epoch']
        self.images_summary_tags = [
            ('train_prediction_sample', [None, self.params.img_height, self.params.img_width * 2, 3]),
            ('val_prediction_sample', [None, self.params.img_height, self.params.img_width * 2, 3])]
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
        if self.args.data_mode == "experiment_h5":
            self.train_data = None
            self.train_data_len = None
            self.val_data = None
            self.val_data_len = None
            self.num_iterations_training_per_epoch = None
            self.num_iterations_validation_per_epoch = None
            self.load_train_data_h5()
            self.generator = self.train_h5_generator
        elif self.args.data_mode == "experiment":
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
            self.num_iterations_testing_per_epoch = None
            self.load_test_data()
            self.generator = self.test_generator
        elif self.args.data_mode == "video":
            self.args.data_mode = "test"
            self.test_data = None
            self.test_data_len = None
            self.num_iterations_testing_per_epoch = None
            self.load_vid_data()
            self.generator = self.test_generator
        elif self.args.data_mode == "overfit":
            self.train_data = None
            self.train_data_len = None
            self.num_iterations_training_per_epoch = None
            self.num_iterations_validation_per_epoch = None
            self.load_overfit_data()
            self.generator = self.overfit_generator
        else:
            print("ERROR Please select a proper data_mode BYE")
            exit(-1)
        ##################################################################################
        # Init metrics class
        self.metrics = Metrics(self.args.num_classes)
        # Init reporter class
        if self.args.mode == 'train' or 'overfit':
            self.reporter = Reporter(self.args.out_dir + 'report_train.json', self.args)
        elif self.args.mode == 'test':
            self.reporter = Reporter(self.args.out_dir + 'report_test.json', self.args)
            ##################################################################################

    @timeit
    def load_overfit_data(self):
        print("Loading data..")
        self.train_data = {'X': np.load(self.args.data_dir + "X_train.npy"),
                           'Y': np.load(self.args.data_dir + "Y_train.npy")}
        self.train_data_len = self.train_data['X'].shape[0] - self.train_data['X'].shape[0] % self.args.batch_size
        self.num_iterations_training_per_epoch = (
                                                     self.train_data_len + self.args.batch_size - 1) // self.args.batch_size
        print("Train-shape-x -- " + str(self.train_data['X'].shape))
        print("Train-shape-y -- " + str(self.train_data['Y'].shape))
        print("Num of iterations in one epoch -- " + str(self.num_iterations_training_per_epoch))
        print("Overfitting data is loaded")

        print("Loading Validation data..")
        self.val_data = self.train_data
        self.val_data_len = self.val_data['X'].shape[0] - self.val_data['X'].shape[0] % self.args.batch_size
        self.num_iterations_validation_per_epoch = (
                                                       self.val_data_len + self.args.batch_size - 1) // self.args.batch_size
        print("Val-shape-x -- " + str(self.val_data['X'].shape) + " " + str(self.val_data_len))
        print("Val-shape-y -- " + str(self.val_data['Y'].shape))
        print("Num of iterations on validation data in one epoch -- " + str(self.num_iterations_validation_per_epoch))
        print("Validation data is loaded")

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
            summary_list = self.sess.run([self.summary_ops[tag] for tag in summaries_dict.keys()],
                                         {self.summary_placeholders[tag]: value for tag, value in
                                          summaries_dict.items()})
            for summary in summary_list:
                self.summary_writer.add_summary(summary, step)
        if summaries_merged is not None:
            self.summary_writer.add_summary(summaries_merged, step)

    @timeit
    def load_train_data(self):
        print("Loading Training data..")
        self.train_data = {'X': np.load(self.args.data_dir + "X_train.npy"),
                           'Y': np.load(self.args.data_dir + "Y_train.npy")}
        self.train_data_len = self.train_data['X'].shape[0]
        self.num_iterations_training_per_epoch = (
                                                     self.train_data_len + self.args.batch_size - 1) // self.args.batch_size
        print("Train-shape-x -- " + str(self.train_data['X'].shape) + " " + str(self.train_data_len))
        print("Train-shape-y -- " + str(self.train_data['Y'].shape))
        print("Num of iterations on training data in one epoch -- " + str(self.num_iterations_training_per_epoch))
        print("Training data is loaded")

        print("Loading Validation data..")
        self.val_data = {'X': np.load(self.args.data_dir + "X_val.npy"),
                         'Y': np.load(self.args.data_dir + "Y_val.npy")}
        self.val_data_len = self.val_data['X'].shape[0] - self.val_data['X'].shape[0] % self.args.batch_size
        self.num_iterations_validation_per_epoch = (
                                                       self.val_data_len + self.args.batch_size - 1) // self.args.batch_size
        print("Val-shape-x -- " + str(self.val_data['X'].shape) + " " + str(self.val_data_len))
        print("Val-shape-y -- " + str(self.val_data['Y'].shape))
        print("Num of iterations on validation data in one epoch -- " + str(self.num_iterations_validation_per_epoch))
        print("Validation data is loaded")

    @timeit
    def load_train_data_h5(self):
        print("Loading Training data..")
        self.train_data = h5py.File(self.args.data_dir + self.args.h5_train_file, 'r')
        self.train_data_len = self.args.h5_train_len
        self.num_iterations_training_per_epoch = (
                                                     self.train_data_len + self.args.batch_size - 1) // self.args.batch_size
        print("Train-shape-x -- " + str(self.train_data['X'].shape) + " " + str(self.train_data_len))
        print("Train-shape-y -- " + str(self.train_data['Y'].shape))
        print("Num of iterations on training data in one epoch -- " + str(self.num_iterations_training_per_epoch))
        print("Training data is loaded")

        print("Loading Validation data..")
        self.val_data = {'X': np.load(self.args.data_dir + "X_val.npy"),
                         'Y': np.load(self.args.data_dir + "Y_val.npy")}
        self.val_data_len = self.val_data['X'].shape[0] - self.val_data['X'].shape[0] % self.args.batch_size
        self.num_iterations_validation_per_epoch = (
                                                       self.val_data_len + self.args.batch_size - 1) // self.args.batch_size
        print("Val-shape-x -- " + str(self.val_data['X'].shape) + " " + str(self.val_data_len))
        print("Val-shape-y -- " + str(self.val_data['Y'].shape))
        print("Num of iterations on validation data in one epoch -- " + str(self.num_iterations_validation_per_epoch))
        print("Validation data is loaded")

    @timeit
    def load_vid_data(self):
        print("Loading Video data..")
        self.test_data = {'X': np.load(self.args.data_dir + "X_vid.npy")}
        self.test_data['Y'] = np.zeros(self.test_data['X'].shape[:3])
        self.test_data_len = self.test_data['X'].shape[0]
        print("Vid-shape-x -- " + str(self.test_data['X'].shape))
        print("Vid-shape-y -- " + str(self.test_data['Y'].shape))
        self.num_iterations_testing_per_epoch = (self.test_data_len + self.args.batch_size - 1) // self.args.batch_size
        print("Video data is loaded")

    @timeit
    def load_test_data(self):
        print("Loading Testing data..")
        self.test_data = {'X': np.load(self.args.data_dir + "X_val.npy"),
                          'Y': np.load(self.args.data_dir + "Y_val.npy")}
        self.test_data_len = self.test_data['X'].shape[0] - self.test_data['X'].shape[0] % self.args.batch_size
        print("Test-shape-x -- " + str(self.test_data['X'].shape))
        print("Test-shape-y -- " + str(self.test_data['Y'].shape))
        self.num_iterations_testing_per_epoch = (self.test_data_len + self.args.batch_size - 1) // self.args.batch_size
        print("Test data is loaded")

    def test_generator(self):
        start = 0
        new_epoch_flag = True
        idx = None
        while True:
            # init index array if it is a new_epoch
            if new_epoch_flag:
                if self.args.shuffle:
                    idx = np.random.choice(self.test_data_len, self.test_data_len, replace=False)
                else:
                    idx = np.arange(self.test_data_len)
                new_epoch_flag = False

            # select the mini_batches
            mask = idx[start:start + self.args.batch_size]
            x_batch = self.test_data['X'][mask]
            y_batch = self.test_data['Y'][mask]

            # update start idx
            start += self.args.batch_size

            if start >= self.test_data_len:
                start = 0
                new_epoch_flag = True

            yield x_batch, y_batch

    def train_generator(self):
        start = 0
        idx = np.random.choice(self.train_data_len, self.num_iterations_training_per_epoch * self.args.batch_size,
                               replace=True)
        while True:
            # select the mini_batches
            mask = idx[start:start + self.args.batch_size]
            x_batch = self.train_data['X'][mask]
            y_batch = self.train_data['Y'][mask]

            # update start idx
            start += self.args.batch_size

            yield x_batch, y_batch

            if start >= self.train_data_len:
                return

    def train_h5_generator(self):
        start = 0
        lenn = self.num_iterations_training_per_epoch * self.args.batch_size
        idx = np.random.choice(self.train_data_len, lenn,
                               replace=True)
        big_batch = lenn // 10
        while True:
            mask = idx[start:start + big_batch]
            x_big_batch = self.train_data['X'][sorted(mask.tolist())]
            y_big_batch = self.train_data['Y'][sorted(mask.tolist())]
            while start != 0 and start % big_batch != 0:
                # select the mini_batches
                x_batch = x_big_batch[start:start + self.args.batch_size]
                y_batch = y_big_batch[start:start + self.args.batch_size]
                # update start idx
                start += self.args.batch_size

                yield x_batch, y_batch

            if start >= self.train_data_len:
                return

    def train(self):
        print("Training mode will begin NOW ..")
        curr_lr = self.model.args.learning_rate
        for cur_epoch in range(self.model.global_epoch_tensor.eval(self.sess) + 1, self.args.num_epochs + 1, 1):

            # init tqdm and get the epoch value
            tt = tqdm(self.generator(), total=self.num_iterations_training_per_epoch,
                      desc="epoch-" + str(cur_epoch) + "-")

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
                             self.model.is_training: True,
                             self.model.curr_learning_rate: curr_lr
                             }

                # Run the feed forward but the last iteration finalize what you want to do
                if cur_iteration < self.num_iterations_training_per_epoch - 1:

                    # run the feed_forward
                    _, loss, acc, summaries_merged = self.sess.run(
                        [self.model.train_op, self.model.loss, self.model.accuracy, self.model.merged_summaries],
                        feed_dict=feed_dict)
                    # log loss and acc
                    loss_list += [loss]
                    acc_list += [acc]
                    # summarize
                    self.add_summary(cur_it, summaries_merged=summaries_merged)

                else:
                    # run the feed_forward
                    _, loss, acc, summaries_merged, segmented_imgs = self.sess.run(
                        [self.model.train_op, self.model.loss, self.model.accuracy, self.model.merged_summaries,
                         self.model.segmented_summary],
                        feed_dict=feed_dict)
                    # log loss and acc
                    loss_list += [loss]
                    acc_list += [acc]
                    total_loss = np.mean(loss_list)
                    total_acc = np.mean(acc_list)
                    # summarize
                    summaries_dict = dict()
                    summaries_dict['train-loss-per-epoch'] = total_loss
                    summaries_dict['train-acc-per-epoch'] = total_acc
                    summaries_dict['train_prediction_sample'] = segmented_imgs
                    self.add_summary(cur_it, summaries_dict=summaries_dict, summaries_merged=summaries_merged)

                    # report
                    self.reporter.report_experiment_statistics('train-acc', 'epoch-' + str(cur_epoch), str(total_acc))
                    self.reporter.report_experiment_statistics('train-loss', 'epoch-' + str(cur_epoch), str(total_loss))
                    self.reporter.finalize()

                    # Update the Global step
                    self.model.global_step_assign_op.eval(session=self.sess,
                                                          feed_dict={self.model.global_step_input: cur_it + 1})

                    # Update the Cur Epoch tensor
                    # it is the last thing because if it is interrupted it repeat this
                    self.model.global_epoch_assign_op.eval(session=self.sess,
                                                           feed_dict={self.model.global_epoch_input: cur_epoch + 1})

                    # print in console
                    tt.close()
                    print("epoch-" + str(cur_epoch))

                # Update the Global step
                self.model.global_step_assign_op.eval(session=self.sess,
                                                      feed_dict={self.model.global_step_input: cur_it + 1})

                # update the cur_iteration
                cur_iteration += 1

            # Save the current checkpoint
            if cur_epoch % self.args.save_every == 0:
                self.save_model()

            # Test the model on validation
            if cur_epoch % self.args.test_every == 0:
                self.test_per_epoch(step=self.model.global_step_tensor.eval(self.sess),
                                    epoch=cur_epoch)

            if cur_epoch % self.args.learning_decay_every == 0:
                curr_lr = curr_lr * self.args.learning_decay
            print('Current learning rate is ', curr_lr)

        print("Training Finished")

    def test_per_epoch(self, step, epoch):
        print("Validation at step:" + str(step) + " at epoch:" + str(epoch) + " ..")

        # init tqdm and get the epoch value
        tt = tqdm(range(self.num_iterations_validation_per_epoch), total=self.num_iterations_validation_per_epoch,
                  desc="Val-epoch-" + str(epoch) + "-")

        # init acc and loss lists
        loss_list = []
        acc_list = []
        inf_list = []

        # idx of minibatch
        idx = 0

        # reset metrics
        self.metrics.reset()

        # get the maximum iou to compare with and save the best model
        max_iou = self.model.best_iou_tensor.eval(self.sess)

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

                start = time.time()
                # run the feed_forward
                out_argmax, loss, acc = self.sess.run(
                    [self.model.out_argmax, self.model.loss, self.model.accuracy],
                    feed_dict=feed_dict)
                end = time.time()
                # log loss and acc
                loss_list += [loss]
                acc_list += [acc]
                inf_list += [end - start]
                # log metrics
                self.metrics.update_metrics_batch(out_argmax, y_batch)

            else:
                start = time.time()
                # run the feed_forward
                out_argmax, loss, acc, segmented_imgs = self.sess.run(
                    [self.model.out_argmax, self.model.loss, self.model.accuracy, self.model.segmented_summary],
                    feed_dict=feed_dict)
                end = time.time()
                # log loss and acc
                loss_list += [loss]
                acc_list += [acc]
                inf_list += [end - start]
                # log metrics
                self.metrics.update_metrics_batch(out_argmax, y_batch)
                # mean over batches
                total_loss = np.mean(loss_list)
                total_acc = np.mean(acc_list)
                mean_iou = self.metrics.compute_final_metrics(self.num_iterations_validation_per_epoch)
                mean_iou_arr = self.metrics.iou
                mean_inference = str(np.mean(inf_list)) + '-seconds'
                # summarize
                summaries_dict = dict()
                summaries_dict['val-loss-per-epoch'] = total_loss
                summaries_dict['val-acc-per-epoch'] = total_acc
                summaries_dict['mean_iou_on_val'] = mean_iou
                summaries_dict['val_prediction_sample'] = segmented_imgs
                self.add_summary(step, summaries_dict=summaries_dict)
                self.summary_writer.flush()

                # report
                self.reporter.report_experiment_statistics('validation-acc', 'epoch-' + str(epoch), str(total_acc))
                self.reporter.report_experiment_statistics('validation-loss', 'epoch-' + str(epoch), str(total_loss))
                self.reporter.report_experiment_statistics('avg_inference_time_on_validation', 'epoch-' + str(epoch),
                                                           str(mean_inference))
                self.reporter.report_experiment_validation_iou('epoch-' + str(epoch), str(mean_iou), mean_iou_arr)
                self.reporter.finalize()

                # print in console
                tt.close()
                print("Val-epoch-" + str(epoch) + "-" + "loss:" + str(total_loss) + "-" +
                      "acc:" + str(total_acc)[:6] + "-mean_iou:" + str(mean_iou))
                print("Last_max_iou: " + str(max_iou))
                if mean_iou > max_iou:
                    print("This validation got a new best iou. so we will save this one")
                    # save the best model
                    self.save_best_model()
                    # Set the new maximum
                    self.model.best_iou_assign_op.eval(session=self.sess,
                                                       feed_dict={self.model.best_iou_input: mean_iou})
                else:
                    print("hmm not the best validation epoch :/..")

                # Break the loop to finalize this epoch
                break

    def test(self):
        print("Testing mode will begin NOW..")

        # load the best model checkpoint to test on it
        self.load_best_model()

        # init tqdm and get the epoch value
        tt = tqdm(range(self.test_data_len))
        naming = np.load(self.args.data_dir + 'names_train.npy')

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

            np.save(self.args.out_dir + 'npy/' + str(cur_iteration) + '.npy', out_argmax[0])
            plt.imsave(self.args.out_dir + 'imgs/' + 'test_' + str(cur_iteration) + '.png', segmented_imgs[0])

            # log loss and acc
            loss_list += [loss]
            acc_list += [acc]

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

        print("Plotting imgs")

    def overfit(self):
        print("Overfitting mode will begin NOW..")

        for cur_epoch in range(self.model.global_epoch_tensor.eval(self.sess) + 1, self.args.num_epochs + 1, 1):

            # init tqdm and get the epoch value
            tt = tqdm(self.generator(), total=self.num_iterations_training_per_epoch,
                      desc="epoch-" + str(cur_epoch) + "-")

            # init the current iterations
            cur_iteration = 0

            # init acc and loss lists
            loss_list = []
            acc_list = []

            # reset metrics
            self.metrics.reset()

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
                    out_argmax, _, loss, acc, summaries_merged = self.sess.run(
                        [self.model.out_argmax, self.model.train_op, self.model.loss, self.model.accuracy,
                         self.model.merged_summaries],
                        feed_dict=feed_dict)

                    # summarize
                    self.add_summary(cur_it, summaries_merged=summaries_merged)

                    # log metrics
                    self.metrics.update_metrics_batch(out_argmax, y_batch)
                    # log loss and acc
                    loss_list += [loss]
                    acc_list += [acc]

                else:

                    # run the feed_forward
                    out_argmax, _, loss, acc, summaries_merged, segmented_imgs = self.sess.run(
                        [self.model.out_argmax, self.model.train_op, self.model.loss, self.model.accuracy,
                         self.model.merged_summaries, self.model.segmented_summary],
                        feed_dict=feed_dict)
                    # summarize
                    summaries_dict = dict()
                    summaries_dict['train_prediction_sample'] = segmented_imgs
                    self.add_summary(cur_it, summaries_dict=summaries_dict, summaries_merged=summaries_merged)

                    # log metrics
                    self.metrics.update_metrics_batch(out_argmax, y_batch)
                    # log loss and acc
                    loss_list += [loss]
                    acc_list += [acc]

                    # Update the Global step
                    self.model.global_step_assign_op.eval(session=self.sess,
                                                          feed_dict={self.model.global_step_input: cur_it + 1})

                    # Update the Cur Epoch tensor
                    # it is the last thing because if it is interrupted it repeat this
                    self.model.global_epoch_assign_op.eval(session=self.sess,
                                                           feed_dict={self.model.global_epoch_input: cur_epoch + 1})

                    mean_iou = self.metrics.compute_final_metrics(self.num_iterations_training_per_epoch)
                    # mean over batches
                    total_loss = np.mean(loss_list)
                    total_acc = np.mean(acc_list)
                    # report
                    self.reporter.report_experiment_statistics('train-acc', 'epoch-' + str(cur_epoch), str(total_acc))
                    self.reporter.report_experiment_statistics('train-loss', 'epoch-' + str(cur_epoch), str(total_loss))
                    self.reporter.finalize()
                    # print in console
                    tt.close()
                    print("epoch-" + str(cur_epoch) + "-" + "loss:" + str(total_loss) + "-" +
                          "acc-" + str(total_acc)[:6] + "  mean-iou: " + str(mean_iou))

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
        self.reporter.finalize()
        self.summary_writer.close()
        self.save_model()
