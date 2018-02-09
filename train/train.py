"""
Trainer class to train Segmentation models
"""

from train.basic_train import BasicTrain
from metrics.metrics import Metrics
from utils.reporter import Reporter
from utils.misc import timeit
from utils.average_meter import FPSMeter

from tqdm import tqdm
import numpy as np
import tensorflow as tf
import matplotlib
import time
import h5py
import pickle
from utils.augmentation import flip_randomly_left_right_image_with_annotation, \
    scale_randomly_image_with_annotation_with_fixed_size_output
import scipy.misc as misc

matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import cv2

from utils.img_utils import decode_labels
from utils.seg_dataloader import SegDataLoader
from tensorflow.contrib.data import Iterator
import pdb
import torchfile

class Train(BasicTrain):
    """
    Trainer class
    """

    def __init__(self, args, sess, train_model, test_model):
        """
        Call the constructor of the base class
        init summaries
        init loading data
        :param args:
        :param sess:
        :param model:
        :return:
        """
        super().__init__(args, sess, train_model, test_model)
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
        if self.args.data_mode == "experiment_tfdata":
            self.data_session = None
            self.train_next_batch, self.train_data_len = self.init_tfdata(self.args.batch_size, self.args.abs_data_dir,
                                                                          (self.args.img_height, self.args.img_width),
                                                                          mode='train')
            self.num_iterations_training_per_epoch = self.train_data_len // self.args.batch_size
            self.generator = self.train_tfdata_generator
        elif self.args.data_mode == "experiment_h5":
            self.train_data = None
            self.train_data_len = None
            self.val_data = None
            self.val_data_len = None
            self.num_iterations_training_per_epoch = None
            self.num_iterations_validation_per_epoch = None
            self.load_train_data_h5()
            self.generator = self.train_h5_generator
        elif self.args.data_mode == "experiment_v2":
            self.targets_resize= self.args.targets_resize
            self.train_data = None
            self.train_data_len = None
            self.val_data = None
            self.val_data_len = None
            self.num_iterations_training_per_epoch = None
            self.num_iterations_validation_per_epoch = None
            self.load_train_data(v2=True)
            self.generator = self.train_generator
        elif self.args.data_mode == "experiment":
            self.train_data = None
            self.train_data_len = None
            self.val_data = None
            self.val_data_len = None
            self.num_iterations_training_per_epoch = None
            self.num_iterations_validation_per_epoch = None
            self.load_train_data()
            self.generator = self.train_generator
        elif self.args.data_mode == "test_tfdata":
            self.test_data = None
            self.test_data_len = None
            self.num_iterations_testing_per_epoch = None
            self.load_test_data()
            self.generator = self.test_tfdata_generator
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
        elif self.args.data_mode == "debug":
            print("Debugging photo loading..")
#            self.debug_x= misc.imread('/data/menna/cityscapes/leftImg8bit/val/lindau/lindau_000048_000019_leftImg8bit.png')
#            self.debug_y= misc.imread('/data/menna/cityscapes/gtFine/val/lindau/lindau_000048_000019_gtFine_labelIds.png')
#            self.debug_x= np.expand_dims(misc.imresize(self.debug_x, (512,1024)), axis=0)
#            self.debug_y= np.expand_dims(misc.imresize(self.debug_y, (512,1024)), axis=0)
            self.debug_x = np.load('data/debug/debug_x.npy')
            self.debug_y = np.load('data/debug/debug_y.npy')
            print("Debugging photo loaded")
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

    def crop(self):
        sh = self.val_data['X'].shape
        temp_val_data = {'X': np.zeros((sh[0] * 2, sh[1], sh[2] // 2, sh[3]), self.val_data['X'].dtype),
                         'Y': np.zeros((sh[0] * 2, sh[1], sh[2] // 2), self.val_data['Y'].dtype)}
        for i in range(sh[0]):
            temp_val_data['X'][i * 2, :, :, :] = self.val_data['X'][i, :, :sh[2] // 2, :]
            temp_val_data['X'][i * 2 + 1, :, :, :] = self.val_data['X'][i, :, sh[2] // 2:, :]
            temp_val_data['Y'][i * 2, :, :] = self.val_data['Y'][i, :, :sh[2] // 2]
            temp_val_data['Y'][i * 2 + 1, :, :] = self.val_data['Y'][i, :, sh[2] // 2:]

        self.val_data = temp_val_data

    def init_tfdata(self, batch_size, main_dir, resize_shape, mode='train'):
        self.data_session = tf.Session()
        print("Creating the iterator for training data")
        with tf.device('/cpu:0'):
            segdl = SegDataLoader(main_dir, batch_size, (resize_shape[0], resize_shape[1]), resize_shape,
                                  # * 2), resize_shape,
                                  'data/cityscapes_tfdata/train.txt')
            iterator = Iterator.from_structure(segdl.data_tr.output_types, segdl.data_tr.output_shapes)
            next_batch = iterator.get_next()

            self.init_op = iterator.make_initializer(segdl.data_tr)
            self.data_session.run(self.init_op)

        print("Loading Validation data in memoryfor faster training..")
        self.val_data = {'X': np.load(self.args.data_dir + "X_val.npy"),
                         'Y': np.load(self.args.data_dir + "Y_val.npy")}
        # self.crop()
        # import cv2
        # cv2.imshow('crop1', self.val_data['X'][0,:,:,:])
        # cv2.imshow('crop2', self.val_data['X'][1,:,:,:])
        # cv2.imshow('seg1', self.val_data['Y'][0,:,:])
        # cv2.imshow('seg2', self.val_data['Y'][1,:,:])
        # cv2.waitKey()

        self.val_data_len = self.val_data['X'].shape[0] - self.val_data['X'].shape[0] % self.args.batch_size
        #        self.num_iterations_validation_per_epoch = (
        #                                                       self.val_data_len + self.args.batch_size - 1) // self.args.batch_size
        self.num_iterations_validation_per_epoch = self.val_data_len // self.args.batch_size

        print("Val-shape-x -- " + str(self.val_data['X'].shape) + " " + str(self.val_data_len))
        print("Val-shape-y -- " + str(self.val_data['Y'].shape))
        print("Num of iterations on validation data in one epoch -- " + str(self.num_iterations_validation_per_epoch))
        print("Validation data is loaded")

        return next_batch, segdl.data_len

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
    def load_train_data(self, v2=False):
        print("Loading Training data..")
        self.train_data = {'X': np.load(self.args.data_dir + "X_train.npy"),
                           'Y': np.load(self.args.data_dir + "Y_train.npy")}
        if v2:
            out_shape= (self.train_data['Y'].shape[1]//self.targets_resize,
                self.train_data['Y'].shape[2]//self.targets_resize)
            yy= np.zeros((self.train_data['Y'].shape[0],out_shape[0],out_shape[1]), dtype=self.train_data['Y'].dtype)
            for y in range(self.train_data['Y'].shape[0]):
                yy[y,...]= misc.imresize(self.train_data['Y'][y,...], out_shape, interp='nearest')
            self.train_data['Y']=yy
        self.train_data_len = self.train_data['X'].shape[0]

        self.num_iterations_training_per_epoch = (self.train_data_len + self.args.batch_size - 1) // self.args.batch_size

        print("Train-shape-x -- " + str(self.train_data['X'].shape) + " " + str(self.train_data_len))
        print("Train-shape-y -- " + str(self.train_data['Y'].shape))
        print("Num of iterations on training data in one epoch -- " + str(self.num_iterations_training_per_epoch))
        print("Training data is loaded")

        print("Loading Validation data..")
        self.val_data = {'X': np.load(self.args.data_dir + "X_val.npy"),
                         'Y': np.load(self.args.data_dir + "Y_val.npy")}
        self.val_data['Y_large']= self.val_data['Y']
        if v2:
            out_shape= (self.val_data['Y'].shape[1]//self.targets_resize,
                self.val_data['Y'].shape[2]//self.targets_resize)
            yy= np.zeros((self.val_data['Y'].shape[0],out_shape[0],out_shape[1]), dtype=self.train_data['Y'].dtype)
            for y in range(self.val_data['Y'].shape[0]):
                yy[y,...]= misc.imresize(self.val_data['Y'][y,...], out_shape, interp='nearest')
            self.val_data['Y']=yy

        self.val_data_len = self.val_data['X'].shape[0] - self.val_data['X'].shape[0] % self.args.batch_size
        self.num_iterations_validation_per_epoch = (self.val_data_len + self.args.batch_size - 1) // self.args.batch_size
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

    def resize(self, data):
        X= []
        Y= []
        for i in range(data['X'].shape[0]):
            X.append(misc.imresize(data['X'][i,...], (self.args.img_height, self.args.img_width)))
            Y.append(misc.imresize(data['Y'][i,...], (self.args.img_height, self.args.img_width), 'nearest'))
        data['X']= np.asarray(X)
        data['Y']= np.asarray(Y)
        return data

    @timeit
    def load_test_data(self):
        print("Loading Testing data..")
        self.test_data = {'X': np.load(self.args.data_dir + "X_val.npy"),
                          'Y': np.load(self.args.data_dir + "Y_val.npy")}
        self.test_data= self.resize(self.test_data)
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

    def train_tfdata_generator(self):
        with tf.device('/cpu:0'):
            while True:
                x_batch, y_batch = self.data_session.run(self.train_next_batch)
                yield x_batch, y_batch[:, :, :, 0]

    def train_h5_generator(self):
        start = 0
        idx = np.random.choice(self.train_data_len, self.train_data_len,
                               replace=False)
        while True:
            # select the mini_batches
            mask = idx[start:start + self.args.batch_size]
            x_batch = self.train_data['X'][sorted(mask.tolist())]
            y_batch = self.train_data['Y'][sorted(mask.tolist())]

            # update start idx
            start += self.args.batch_size

            if start >= self.train_data_len:
                return

            yield x_batch, y_batch

    def train(self):
        print("Training mode will begin NOW ..")
        #curr_lr= self.model.args.learning_rate
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
                             self.model.is_training: True
#                             self.model.curr_learning_rate:curr_lr
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
#                    self.add_summary(cur_it, summaries_merged=summaries_merged)

                else:
                    # run the feed_forward
                    _, loss, acc, summaries_merged, segmented_imgs = self.sess.run(
                        [self.model.train_op, self.model.loss, self.model.accuracy,
                         self.model.merged_summaries, self.model.segmented_summary],
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
#                    self.add_summary(cur_it, summaries_dict=summaries_dict, summaries_merged=summaries_merged)

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
                    print("epoch-" + str(cur_epoch) + "-" + "loss:" + str(total_loss) + "-" + " acc:" + str(total_acc)[
                                                                                                        :6])

                    # Break the loop to finalize this epoch
                    break


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
                                    epoch=self.model.global_epoch_tensor.eval(self.sess))
#            if cur_epoch % self.args.learning_decay_every == 0:
#                curr_lr= curr_lr*self.args.learning_decay
#                print('Current learning rate is ', curr_lr)

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
            if self.args.data_mode == 'experiment_v2':
                y_batch_large= self.val_data['Y_large'][idx:idx+self.args.batch_size]

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
                out_argmax, loss, acc, summaries_merged = self.sess.run(
                    [self.model.out_argmax, self.model.loss, self.model.accuracy, self.model.merged_summaries],
                    feed_dict=feed_dict)
                end = time.time()
                # log loss and acc
                loss_list += [loss]
                acc_list += [acc]
                inf_list += [end - start]
                if self.args.data_mode == 'experiment_v2':
                    yy= np.zeros((out_argmax.shape[0], y_batch_large.shape[1], y_batch_large.shape[2]), dtype=np.uint32)
                    out_argmax= np.asarray(out_argmax, dtype= np.uint8)
                    for y in range(out_argmax.shape[0]):
                        yy[y,...]= misc.imresize(out_argmax[y,...], y_batch_large.shape[1:], interp='nearest')
                    y_batch= y_batch_large
                    out_argmax= yy

                # log metrics
                self.metrics.update_metrics_batch(out_argmax, y_batch)

            else:
                start = time.time()
                # run the feed_forward
                if self.args.data_mode=='experiment_v2': #Issues in concatenating gt and img with diff sizes now for segmented_imgs
                     out_argmax, acc = self.sess.run(
                        [self.test_model.out_argmax, self.test_model.accuracy],
                        feed_dict=feed_dict)
                else:
                     out_argmax, acc, segmented_imgs = self.sess.run(
                        [self.test_model.out_argmax, self.test_model.accuracy, self.test_model.segmented_summary],feed_dict=feed_dict)

                end = time.time()
                # log loss and acc
                acc_list += [acc]
                inf_list += [end - start]
                # log metrics
                self.metrics.update_metrics_batch(out_argmax, y_batch)
                # mean over batches
                total_acc = np.mean(acc_list)
                mean_iou = self.metrics.compute_final_metrics(self.num_iterations_validation_per_epoch)
                mean_iou_arr = self.metrics.iou
                mean_inference = str(np.mean(inf_list)) + '-seconds'
                # summarize
                summaries_dict = dict()
                summaries_dict['val-acc-per-epoch'] = total_acc
                summaries_dict['mean_iou_on_val'] = mean_iou
                summaries_dict['val_prediction_sample'] = segmented_imgs
#                self.add_summary(step, summaries_dict=summaries_dict, summaries_merged=summaries_merged)

                # report
                self.reporter.report_experiment_statistics('validation-acc', 'epoch-' + str(epoch), str(total_acc))
                self.reporter.report_experiment_statistics('avg_inference_time_on_validation', 'epoch-' + str(epoch), str(mean_inference))
                self.reporter.report_experiment_validation_iou('epoch-' + str(epoch), str(mean_iou), mean_iou_arr)
                self.reporter.finalize()

                # print in console
                tt.close()
                print("Val-epoch-" + str(epoch) + "-"+
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
                break

                # Break the loop to finalize this epoch

    def linknet_postprocess(self, gt):
        gt2 = gt - 1
        gt2[gt == -1] = 19
        return gt2

    def test(self, pkl=False):
        print("Testing mode will begin NOW..")

        # load the best model checkpoint to test on it
        if not pkl:
            self.load_best_model()

        # init tqdm and get the epoch value
        tt = tqdm(range(self.test_data_len))

        # init acc and loss lists
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
            idx += 1

            # Feed this variables to the network
            feed_dict = {self.test_model.x_pl: x_batch,
                         self.test_model.y_pl: y_batch,
                         self.test_model.is_training: False
                         }

            # run the feed_forward
            out_argmax, segmented_imgs = self.sess.run([self.test_model.out_argmax,
                self.test_model.segmented_summary],
                feed_dict=feed_dict)

            #print('mean preds ', out_argmax.mean())
            # np.save(self.args.out_dir + 'npy/' + str(cur_iteration) + '.npy', out_argmax[0])
            plt.imsave(self.args.out_dir + 'imgs/' + 'test_' + str(cur_iteration) + '.png', segmented_imgs[0])
            self.metrics.update_metrics(out_argmax[0], y_batch[0], 0, 0)

        # mean over batches
        mean_iou = self.metrics.compute_final_metrics(self.test_data_len)

        # print in console
        tt.close()
        print("Here the statistics")
        print("mean_iou: " + str(mean_iou))
        print("foreground iou: " + str(self.metrics.iou[1]))

        print("Plotting imgs")
        for i in range(len(img_list)):
            plt.imsave(self.args.imgs_dir + 'test_' + str(i) + '.png', img_list[i])

    def test_inference(self):
        """
        Like the testing function but this one is for calculate the inference time
        and measure the frame per second
        """
        print("INFERENCE mode will begin NOW..")

        # load the best model checkpoint to test on it
        self.load_best_model()

        # init tqdm and get the epoch value
        tt = tqdm(range(self.test_data_len))

        # idx of image
        idx = 0

        # create the FPS Meter
        fps_meter = FPSMeter()

        # loop by the number of iterations
        for cur_iteration in tt:
            # load mini_batches
            x_batch = self.test_data['X'][idx:idx + 1]
            y_batch = self.test_data['Y'][idx:idx + 1]

            # update idx of mini_batch
            idx += 1

            # Feed this variables to the network
            if self.args.random_cropping:
                feed_dict = {self.test_model.x_pl_before: x_batch,
                             self.test_model.y_pl_before: y_batch,
                             self.test_model.is_training: False,
                             }
            else:
                feed_dict = {self.test_model.x_pl: x_batch,
                             self.test_model.y_pl: y_batch,
                             self.test_model.is_training: False
                             }

            # calculate the time of one inference
            start = time.time()

            # run the feed_forward
            _ = self.sess.run(
                [self.test_model.out_argmax],
                feed_dict=feed_dict)

            # update the FPS meter
            fps_meter.update(time.time() - start)

        fps_meter.print_statistics()

    def finalize(self):
        self.reporter.finalize()
        self.summary_writer.close()
        self.save_model()

    def debug_layers(self):
        """
        This function will be responsible for output all outputs of all layers and dump them in a pickle

        :return:
        """
        print("Debugging mode will begin NOW..")

        layers = tf.get_collection('debug_layers')
        print("ALL Layers in the collection that i wanna to run {} layer".format(len(layers)))
        for layer in layers:
            print(layer)

        # exit(0)

        # reset metrics
        self.metrics.reset()

        print('mean image ', self.debug_x.mean())
        print('mean gt ', self.debug_y.mean())

        self.debug_y = self.linknet_preprocess_gt(self.debug_y)

        feed_dict = {self.test_model.x_pl: self.debug_x,
                     self.test_model.y_pl: self.debug_y,
                     self.test_model.is_training: False
                     }

#        var = [v for v in tf.all_variables() if v.op.name == "network/decoder_block_4/deconv/deconv/weights"]
#        conv_w= self.sess.run(var[0])
#        var = [v for v in tf.all_variables() if v.op.name == "network/decoder_block_4/deconv/deconv/biases"]
#        bias= self.sess.run(var[0])

        # run the feed_forward
        out_layers = self.sess.run(layers, feed_dict=feed_dict)
        for layer in out_layers:
            print(layer.shape)

#        dict_out= torchfile.load('out_networks_layers/dict_out.t7')
##        init= tf.constant_initializer(conv_w)
##        conv_w1 = tf.get_variable('my_weights', [3,3,128,128], tf.float32, initializer=init, trainable=True)
#        pp= tf.nn.relu(layers[39])
#        out_relu= self.sess.run(pp, feed_dict={self.test_model.x_pl: self.debug_x,
#                     self.test_model.y_pl: self.debug_y,
#                     self.test_model.is_training: False
#                     })
##        pp = tf.nn.conv2d_transpose(layers[39], conv_w, (1,32,64,128), strides=(1,2,2,1), padding="SAME")
##        pp= tf.image.resize_images(layers[39], (32,64))
##        pp = tf.nn.conv2d(pp, conv_w, strides=(1,1,1,1), padding="SAME")
##        bias1= tf.get_variable('my_bias', 128, tf.float32, tf.constant_initializer(bias))
#        pp = tf.nn.bias_add(pp, bias)
#        #self.sess.run(conv_w1.initializer)
#        #self.sess.run(bias1.initializer)
#        out_deconv= self.sess.run(pp, feed_dict={self.test_model.x_pl: self.debug_x,
#                     self.test_model.y_pl: self.debug_y,
#                     self.test_model.is_training: False
#                     })
#        out_deconv_direct= self.sess.run(layers[40], feed_dict={self.test_model.x_pl: self.debug_x,
#                     self.test_model.y_pl: self.debug_y,
#                     self.test_model.is_training: False
#                     })
#        pdb.set_trace()

        # print(out_layers)
        # exit(0)

        # dump them in a pickle
        with open("out_networks_layers/out_linknet_layers.pkl", "wb") as f:
            pickle.dump(out_layers, f, protocol=2)

        # run the feed_forward again to see argmax and segmented
        out_argmax, segmented_imgs = self.sess.run(
            [self.test_model.out_argmax,
             self.test_model.segmented_summary],
            feed_dict=feed_dict)

        print('mean preds ', out_argmax[0].mean())

        plt.imsave(self.args.out_dir + 'imgs/' + 'debug.png', segmented_imgs[0])

        self.metrics.update_metrics(out_argmax[0], self.debug_y, 0, 0)

        mean_iou = self.metrics.compute_final_metrics(1)

        print("mean_iou_of_debug: " + str(mean_iou))
