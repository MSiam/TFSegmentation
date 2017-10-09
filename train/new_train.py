"""
New trainer faster than ever
"""

from metrics.metrics import Metrics
from utils.reporter import Reporter
from utils.misc import timeit

from tqdm import tqdm
import numpy as np
import tensorflow as tf
import matplotlib
import time

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class NewTrain(object):
    def __init__(self, args, sess, model):
        print("\nTraining is initializing itself\n")

        self.args = args
        self.sess = sess
        self.model = model

        # shortcut for model params
        self.params = self.model.params

        # To initialize all variables
        self.init = None
        self.init_model()

        # Create a saver object
        self.saver = tf.train.Saver(max_to_keep=self.args.max_to_keep,
                                    keep_checkpoint_every_n_hours=10,
                                    save_relative_paths=True)

        self.saver_best = tf.train.Saver(max_to_keep=1,
                                         save_relative_paths=True)

        # Load from latest checkpoint if found
        self.load_model()

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
        if self.args.mode == 'train':
            self.num_iterations_training_per_epoch = self.args.tfrecord_train_len // self.args.batch_size
            self.num_iterations_validation_per_epoch = self.args.tfrecord_val_len // self.args.batch_size
        else:
            self.test_data = None
            self.test_data_len = None
            self.num_iterations_testing_per_epoch = None
            self.load_test_data()
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
    def load_test_data(self):
        print("Loading Testing data..")
        self.test_data = {'X': np.load(self.args.data_dir + "X_val.npy"),
                          'Y': np.load(self.args.data_dir + "Y_val.npy")}
        self.test_data_len = self.test_data['X'].shape[0] - self.test_data['X'].shape[0] % self.args.batch_size
        print("Test-shape-x -- " + str(self.test_data['X'].shape))
        print("Test-shape-y -- " + str(self.test_data['Y'].shape))
        self.num_iterations_testing_per_epoch = (self.test_data_len + self.args.batch_size - 1) // self.args.batch_size
        print("Test data is loaded")

    @timeit
    def init_model(self):
        print("Initializing the variables of the model")
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)
        print("Initialization finished")

    def save_model(self):
        """
        Save Model Checkpoint
        :return:
        """
        print("saving a checkpoint")
        self.saver.save(self.sess, self.args.checkpoint_dir, self.model.global_step_tensor)
        print("Saved a checkpoint")

    def save_best_model(self):
        """
        Save BEST Model Checkpoint
        :return:
        """
        print("saving a checkpoint for the best model")
        self.saver_best.save(self.sess, self.args.checkpoint_best_dir, self.model.global_step_tensor)
        print("Saved a checkpoint for the best model")

    def load_best_model(self):
        """
        Load the best model checkpoint
        :return:
        """
        print("loading a checkpoint for BEST ONE")
        latest_checkpoint = tf.train.latest_checkpoint(self.args.checkpoint_best_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver_best.restore(self.sess, latest_checkpoint)
        else:
            print("ERROR NO best checkpoint found")
            exit(-1)
        print("BEST MODEL LOADED..")

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
    def load_model(self):
        """
        Load the latest checkpoint
        :return:
        """
        try:
            # This is for loading the pretrained weights if they can't be loaded during initialization.
            self.model.encoder.load_pretrained_weights(self.sess)
        except AttributeError:
            pass

        print("Searching for a checkpoint")
        latest_checkpoint = tf.train.latest_checkpoint(self.args.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)
            print("Model loaded from the latest checkpoint\n")
        else:
            print("\n.. No ckpt, SO First time to train :D ..\n")

    def train(self):
        print("Training mode will begin NOW ..")
        tf.train.start_queue_runners()
        curr_lr = self.model.args.learning_rate
        for cur_epoch in range(self.model.global_epoch_tensor.eval(self.sess) + 1, self.args.num_epochs + 1, 1):

            # init tqdm and get the epoch value
            tt = tqdm(range(self.num_iterations_training_per_epoch), total=self.num_iterations_training_per_epoch,
                      desc="epoch-" + str(cur_epoch) + "-")

            # init acc and loss lists
            loss_list = []
            acc_list = []

            # loop by the number of iterations
            for cur_iteration in tt:

                # get the cur_it for the summary
                cur_it = self.model.global_step_tensor.eval(self.sess)

                # Feed this variables to the network
                feed_dict = {
                    self.model.handle: self.model.training_handle,
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
                    print("epoch-" + str(cur_epoch) + "-" + "loss:" + str(total_loss) + "-" + " acc:" + str(total_acc)[
                                                                                                        :6])

                    # Break the loop to finalize this epoch
                    break

                # Update the Global step
                self.model.global_step_assign_op.eval(session=self.sess,
                                                      feed_dict={self.model.global_step_input: cur_it + 1})

            # Save the current checkpoint
            if cur_epoch % self.args.save_every == 0:
                self.save_model()

            # Test the model on validation
            if cur_epoch % self.args.test_every == 0:
                self.test_per_epoch(step=self.model.global_step_tensor.eval(self.sess),
                                    epoch=self.model.global_epoch_tensor.eval(self.sess))

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

        # reset metrics
        self.metrics.reset()

        # get the maximum iou to compare with and save the best model
        max_iou = self.model.best_iou_tensor.eval(self.sess)

        # init dataset to validation
        self.sess.run(self.model.validation_iterator.initializer)

        # loop by the number of iterations
        for cur_iteration in tt:
            # Feed this variables to the network
            feed_dict = {
                self.model.handle: self.model.validation_handle,
                self.model.is_training: False
            }

            # Run the feed forward but the last iteration finalize what you want to do
            if cur_iteration < self.num_iterations_validation_per_epoch - 1:

                start = time.time()
                # run the feed_forward
                next_img, out_argmax, loss, acc = self.sess.run(
                    [self.model.next_img, self.model.out_argmax, self.model.loss, self.model.accuracy],
                    feed_dict=feed_dict)
                end = time.time()
                # log loss and acc
                loss_list += [loss]
                acc_list += [acc]
                inf_list += [end - start]
                # log metrics
                self.metrics.update_metrics_batch(out_argmax, next_img[1])

            else:
                start = time.time()
                # run the feed_forward
                next_img, out_argmax, loss, acc, segmented_imgs = self.sess.run(
                    [self.model.next_img, self.model.out_argmax, self.model.loss, self.model.accuracy,
                     self.model.segmented_summary],
                    feed_dict=feed_dict)
                end = time.time()
                # log loss and acc
                loss_list += [loss]
                acc_list += [acc]
                inf_list += [end - start]
                # log metrics
                self.metrics.update_metrics_batch(out_argmax, next_img[1])
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

    def finalize(self):
        self.reporter.finalize()
        self.summary_writer.close()
        self.save_model()
