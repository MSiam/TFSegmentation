import tensorflow as tf
import numpy as np
import pickle
from utils import Constants
from utils.Measures import create_confusion_matrix
from utils import Measures
import glob
from tensorflow.python.training import moving_averages
from collections import namedtuple
from datasets.Loader import load_dataset
from scipy.ndimage import imread
import os
import numpy
from tqdm import tqdm
from utils.one_shot_utils import adjust_results_to_targets, process_forward_result, flip_if_necessary, average_measures
from models.model import Onavos
import pdb
from Forwarding.OnlineAdaptingForwarder import OnlineAdaptingForwarder
class Onavos_1stream(Onavos):
    def __init__(self, sess, config):
        super(Onavos_1stream, self).__init__(sess, config)
        self.config = config
        self.sess = sess
        self.coordinator = tf.train.Coordinator()
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        # load the 2 datasets
        self.prepare_datasets()
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=1e-4)
        self.void_labels=self.valid_data.void_label()
        if self.config.task=="train" or self.config.task=="online":
            self.train_net = self.build(self.train_inputs,self.train_labels,False)
        self.test_net = self.build(self.valid_inputs, self.valid_labels, (config.task == "train" or self.config.task=="online"))

        self.saver = tf.train.Saver(max_to_keep=2)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        if self.config.load_model:
            self.load()
        # self.loss_summed= self.regularizer + self.loss



    def prepare_datasets(self):
        if self.config.task == "train" or self.config.task=="online":
            self.train_data = load_dataset(self.config, "train", self.sess, self.coordinator)
            inputs_tensors_dict = self.train_data.create_input_tensors_dict(self.config.batch_size)
            self.train_inputs = inputs_tensors_dict["inputs"]
            self.train_labels = inputs_tensors_dict["labels"]
            self.train_raw_labels = inputs_tensors_dict.get("raw_labels", None)
            self.train_index_imgs = inputs_tensors_dict.get("index_imgs", None)
            self.train_tags = inputs_tensors_dict.get("tags")

        self.valid_data = load_dataset(self.config, "valid", self.sess, self.coordinator)
        inputs_tensors_dict = self.valid_data.create_input_tensors_dict(self.config.batch_size)
        self.valid_inputs = inputs_tensors_dict["inputs"]
        self.valid_labels = inputs_tensors_dict["labels"]
        self.valid_raw_labels = inputs_tensors_dict.get("raw_labels", None)
        self.valid_index_imgs = inputs_tensors_dict.get("index_imgs", None)
        self.valid_tags = inputs_tensors_dict.get("tags")

        tf.train.start_queue_runners(self.sess)

    def save(self):
        self.saver.save(self.sess, self.config.checkpoint_dir+"chk_point")
        print("Model saved")

    def load(self):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint==None:
            "ERROR Loading the model"
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)
            print("Model loaded")

    # model
    def build(self, inputs,labels,reuse):
        with tf.variable_scope("",reuse=reuse):

            conv0 = tf.layers.conv2d(inputs, 64, 3, use_bias=False, padding='SAME', name='conv0')

            res0 = self.residual_unit2('res0', conv0, n_features=128, strides=[[2, 2], [1, 1]])
            res1 = self.residual_unit2('res1', res0)
            res2 = self.residual_unit2('res2', res1)

            res3 = self.residual_unit2('res3', res2, n_features=256, strides=[[2, 2], [1, 1]])
            res4 = self.residual_unit2('res4', res3)
            res5 = self.residual_unit2('res5', res4)

            res6 = self.residual_unit2('res6', res5, n_features=512, strides=[[2, 2], [1, 1]])
            res7 = self.residual_unit2('res7', res6)
            res8 = self.residual_unit2('res8', res7)
            res9 = self.residual_unit2('res9', res8)
            res10 = self.residual_unit2('res10', res9)
            res11 = self.residual_unit2('res11', res10)

            res12 = self.residual_unit2('res12', res11, n_features=[512, 1024], dilations=[1, 2])
            res13 = self.residual_unit2('res13', res12, n_features=[512, 1024], dilations=[2, 2])
            res14 = self.residual_unit2('res14', res13, n_features=[512, 1024], dilations=[2, 2])

            res15 = self.residual_unit2('res15', res14, n_convs=3, n_features=[512, 1024, 2048],
                                        filter_size=[[1, 1], [3, 3], [1, 1]], dilations=[1, 4, 1])

            res16 = self.residual_unit2('res16', res15, n_convs=3, n_features=[1024, 2048, 4096],
                                        filter_size=[[1, 1], [3, 3], [1, 1]], dilations=[1, 4, 1])

            conv1 = tf.layers.batch_normalization(res16, axis=-1, momentum=0.95, epsilon=1e-5,
                                                  training=self.is_training, name='conv1/bn')
            conv1 = tf.layers.conv2d(tf.nn.relu(conv1), 512, 3, dilation_rate=12,
                                     kernel_regularizer=self.regularizer,
                                     padding='SAME', name='conv1')
            conv1 = tf.nn.relu(conv1)

            out, pred, measures, loss = self.segmentation_softmax('output', conv1, labels, 2, self.void_labels, filter_size=(3, 3),
                                                                  input_activation=tf.nn.relu,
                                                                  dilation=12, resize_targets=True, loss="bootstrapped_ce",
                                                                  fraction=0.25)
            ret = {}
            ret['out'] = out
            ret['pred'] = pred
            ret['measures'] = measures
            ret['loss'] = loss
        return ret

    def residual_unit2(self, name, inputs, n_convs=2, n_features=None, dilations=None, strides=None,
                       filter_size=None, activation=tf.nn.relu, batch_norm_decay=0.95):
        n_features_inp = int(inputs.get_shape()[-1])
        if dilations is not None:
            assert strides is None
        elif strides is None:
            strides = [[1, 1]] * n_convs
        if filter_size is None:
            filter_size = [[3, 3]] * n_convs
        if n_features is None:
            n_features = n_features_inp
        if not isinstance(n_features, list):
            n_features = [n_features] * n_convs
        with tf.variable_scope(name):
            curr = tf.layers.batch_normalization(inputs, axis=-1, momentum=batch_norm_decay, epsilon=1e-5,
                                                 training=self.is_training, name='bn0')
            if activation is not None:
                curr = activation(curr)

            if strides is None:
                strides_res = [1, 1]
            else:
                strides_res = np.prod(strides, axis=0).tolist()

            if (n_features[-1] != n_features_inp) or (strides_res != [1, 1]):
                if dilations is None:
                    res = tf.layers.conv2d(curr, n_features[-1], 1, strides=strides_res,
                                           kernel_regularizer=self.regularizer,
                                           use_bias=False, padding='SAME', name='conv0')
                else:
                    res = tf.layers.conv2d(curr, n_features[-1], 1, kernel_regularizer=self.regularizer,
                                           use_bias=False, padding='SAME', name='conv0')
            else:
                res = inputs
            if dilations is None:
                curr = tf.layers.conv2d(curr, n_features[0], filter_size[0], strides=strides[0],
                                        kernel_regularizer=self.regularizer, use_bias=False, padding='SAME',
                                        name='conv1')
            else:
                curr = tf.layers.conv2d(curr, n_features[0], filter_size[0], dilation_rate=dilations[0],
                                        kernel_regularizer=self.regularizer, use_bias=False, padding='SAME',
                                        name='conv1')

            for idx in range(1, n_convs):
                curr = tf.layers.batch_normalization(curr, axis=-1, momentum=batch_norm_decay, epsilon=1e-5,
                                                     training=self.is_training, name='bn' + str(idx))

                if activation is not None:
                    curr = activation(curr)

                if dilations is None:
                    curr = tf.layers.conv2d(curr, n_features[idx], filter_size[idx], strides=strides[idx],
                                            kernel_regularizer=self.regularizer, use_bias=False, padding='SAME',
                                            name='conv' + str(idx + 1))
                else:
                    curr = tf.layers.conv2d(curr, n_features[idx], filter_size[idx], dilation_rate=dilations[idx],
                                            kernel_regularizer=self.regularizer, use_bias=False, padding='SAME',
                                            name='conv' + str(idx + 1))
            curr = curr + res
            return curr

    def segmentation_softmax(self, name, inp, targets, n_classes, void_label, filter_size=(1, 1),
                             input_activation=None, dilation=None, resize_targets=False, resize_logits=False, loss="ce",
                             fraction=None):
        n_features_inp = int(inp.get_shape()[-1])
        filter_size = list(filter_size)

        with tf.variable_scope(name):
            if input_activation is not None:
                inp = input_activation(inp)

            if dilation is None:
                y_pred = tf.layers.conv2d(inp, n_classes, filter_size[0], kernel_regularizer=self.regularizer,
                                          padding='SAME', name='conv')
            else:
                y_pred = tf.layers.conv2d(inp, n_classes, filter_size[0], kernel_regularizer=self.regularizer,
                                          dilation_rate=dilation, padding='SAME', name='conv')
            outputs = tf.nn.softmax(y_pred, -1, 'softmax')

            if resize_targets:
                targets = tf.image.resize_nearest_neighbor(targets, tf.shape(y_pred)[1:3])
            if resize_logits:
                y_pred = tf.image.resize_images(y_pred, tf.shape(targets)[1:3])
            pred = tf.argmax(y_pred, axis=-1)

            targets = tf.cast(targets, tf.int64)
            targets = tf.squeeze(targets, axis=3)

            # TODO: Void label is not considered in the iou calculation.
            if void_label is not None:
                # avoid nan by replacing void label by 0
                # note: the loss for these cases is multiplied by 0 below
                void_label_mask = tf.equal(targets, void_label)
                no_void_label_mask = tf.logical_not(void_label_mask)
                targets = tf.where(void_label_mask, tf.zeros_like(targets), targets)
            else:
                no_void_label_mask = None
            measures = self.create_measures(pred, targets)
            loss = self.create_loss(loss, fraction, no_void_label_mask, targets, void_label, y_pred)
        return outputs, pred, measures, loss

    def bootstrapped_ce_loss(self, ce, fraction):
        # only consider k worst pixels (lowest posterior probability) per image
        assert fraction is not None
        batch_size = ce.get_shape().as_list()[0]
        if batch_size is None:
            batch_size = tf.shape(ce)[0]
        k = tf.cast(tf.cast(tf.shape(ce)[1] * tf.shape(ce)[2], tf.float32) * fraction, tf.int32)
        bs_ce, _ = tf.nn.top_k(tf.reshape(ce, shape=[batch_size, -1]), k=k, sorted=False)
        bs_ce = tf.reduce_mean(bs_ce, axis=1)
        bs_ce = tf.reduce_sum(bs_ce, axis=0)
        return bs_ce

    def create_loss(self, loss_str, fraction, no_void_label_mask, targets, void_label, y_pred):
        ce = None
        if "ce" in loss_str:
            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=targets, name="ce")

            if  void_label is not None:
                mask = tf.cast(no_void_label_mask, tf.float32)
                ce *= mask
        if loss_str == "ce":
            ce = tf.reduce_mean(ce, axis=[1, 2])
            ce = tf.reduce_sum(ce, axis=0)
            loss = ce
        elif loss_str == "bootstrapped_ce":
            bs_ce = self.bootstrapped_ce_loss(ce, fraction)
            loss = bs_ce
        else:
            assert False, "Unknown loss " + loss_str
        return loss

    def create_measures(self, pred, targets, n_classes=2):
        measures = {}
        conf_matrix = tf.py_func(create_confusion_matrix, [pred, targets, n_classes], [tf.int64])
        measures[Constants.CONFUSION_MATRIX] = conf_matrix[0]
        return measures



    def parse_onavos_weights(self, model_path):
        with open(model_path, 'rb') as output:
            loaded_weights = pickle.load(output, encoding='latin1')
            loaded_model_names = []

            for name, v in loaded_weights.items():
                if name == 'global_step:0':
                    continue
                loaded_model_names.append(name)
            model_weights = {}
            model_names = []

            model_trainables_vars = tf.all_variables()
            for i in range(len(model_trainables_vars)):
                if  model_trainables_vars[i].name == 'global_step:0':
                    continue
                name = model_trainables_vars[i].name
                name = name.replace('kernel', 'W')
                name = name.replace('bn', 'zbn')
                model_names.append(name)
                model_weights[model_trainables_vars[i].name] = model_trainables_vars[i]

            # pdb.set_trace()
            loaded_model_names = sorted(loaded_model_names)
            model_names = sorted(model_names)

            for i in range(len(loaded_model_names)):
                print(i, ' loaded ', loaded_model_names[i], ' in ', model_names[i])
                name = model_names[i]
                # if i !=0:
                name = name.replace('W', 'kernel')
                name = name.replace('zbn', 'bn')
                assign_op = model_weights[name].assign(loaded_weights[loaded_model_names[i]])
                self.sess.run(assign_op)
        self.save()
        print("loading weights done")

    def single_run(self, run_list, feed_dict):
        ys_argmax_val, targets_val, tags_val, logits_val = self.sess.run(run_list, feed_dict)
        ys_argmax_val = numpy.expand_dims(ys_argmax_val, axis=3)
        return ys_argmax_val, targets_val, tags_val, logits_val

    def multiple_run(self, run_list, feed_dict):
        accumulator, index_img, targets_val, tags_val = self.sess.run(run_list, feed_dict)
        accumulator = flip_if_necessary(accumulator, index_img)
        results = run_list[0]
        for k in range(self.config.n_test_samples - 1):
            ys_val, index_img = self.sess.run([results, self.valid_index_imgs], feed_dict)
            ys_val = flip_if_necessary(ys_val, index_img)
            accumulator += ys_val

        logits = accumulator / self.config.n_test_samples
        ys_argmax_val = numpy.expand_dims(numpy.argmax(logits, axis=-1), axis=3)
        return ys_argmax_val, logits, targets_val, tags_val



    def one_shot_evaluation(self):
        f = open(self.config.logfile_path, 'w+')
        f.write("evaluation logging")

        results = adjust_results_to_targets(self.test_net['out'], self.valid_raw_labels)
        ys_argmax = tf.argmax(results, 3)

        video_ids = range(0, self.valid_data.n_videos())
        all_measure = []
        for idx in video_ids:
            measures = []
            self.valid_data.set_video_idx(idx)

            n_total = self.valid_data.num_examples_per_epoch()
            for i in tqdm(range(n_total)):

                feed_dict = self.valid_data.feed_dict_for_video_frame(frame_idx=i, with_annotations=True)
                feed_dict[self.is_training] = False
                if self.config.n_test_samples <= 1:
                    ys_argmax_val, targets_val, tags_val, logits_val = self.single_run(
                        [ys_argmax, self.valid_raw_labels, self.valid_tags, self.test_net['out']], feed_dict)
                else:
                    ys_argmax_val, logits_val, targets_val, tags_val = self.multiple_run(
                        [results, self.valid_index_imgs, self.valid_raw_labels, self.valid_tags], feed_dict)

                for y_argmax, logit, target, tag in zip(ys_argmax_val, logits_val, targets_val, tags_val):
                    measure = process_forward_result(y_argmax, logit, target, tag)
                    print(measure)

                    measures.append(measure)
            if self.config.ignore_first_and_last_results:
                measures = measures[1:-1]
            all_measure += [average_measures(measures)]
            print("seqeunce ", idx, " : ", average_measures(measures))
            f.write(str("seqeunce " + str(idx) + " : " + str(average_measures(measures))))
            f.flush()

        print("All data avg  ", " : ", average_measures(all_measure))
        f.write(str("All data avg  : " + str(average_measures(all_measure))))
        f.flush()

        f.close()

    def online_forward(self, sess,config,model,trainer):
        forwarder = OnlineAdaptingForwarder(sess,config,model,trainer)
        forwarder.forward(None, None,self.config.save_results, self.config.save_logits)

    def test_on_davis(self):
        measures_accumulated = {}
        # for img_num in range(self.valid_data.num_examples_per_epoch()):
        for img_num in range(50):
            feed_dict = {self.is_training: False}
            measure = self.sess.run([self.measures], feed_dict)
            measures_accumulated = Measures.calc_measures_sum(measures_accumulated, measure[0])
            current_iou = Measures.calc_measures_avg(measure[0], 1, [255])

            print('image ' + str(img_num + 1) + ' finished')
            avg_iou = Measures.calc_measures_avg(measures_accumulated, img_num + 1, [255])
            print('current_image iou : ', current_iou, 'avg iou : ', avg_iou)

        measures_accumulated = Measures.calc_measures_avg(measures_accumulated,
                                                          self.train_data.num_examples_per_epoch(), [255])
        return measures_accumulated

