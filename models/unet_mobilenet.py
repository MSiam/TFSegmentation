from models.basic.basic_model import BasicModel
from models.encoders.mobilenet import MobileNet
from layers.convolution import conv2d_transpose, conv2d

import tensorflow as tf


class UNetMobileNet(BasicModel):
    def __init__(self, args, phase=0):
        super().__init__(args, phase=phase)
        # init encoder
        self.encoder = None

    def build(self):
        print("\nBuilding the MODEL...")
        self.init_input()
        self.init_network()
        self.init_output()
        self.init_train()
        self.init_summaries()
        print("The Model is built successfully\n")

    @staticmethod
    def _debug(operation):
        print("Layer_name: " + operation.op.name + " -Output_Shape: " + str(operation.shape.as_list()))

    def init_network(self):
        """
        Building the Network here
        :return:
        """

        # Init MobileNet as an encoder
        self.encoder = MobileNet(x_input=self.x_pl, num_classes=self.params.num_classes,
                                 pretrained_path=self.args.pretrained_path,
                                 train_flag=self.is_training, width_multipler=1.0, weight_decay=self.args.weight_decay)

        # Build Encoding part
        self.encoder.build()

        # Build Decoding part
        with tf.name_scope('upscale_1'):
            self.expand11 = conv2d('expand1_1', x=self.encoder.conv5_6, batchnorm_enabled=True, is_training= self.is_training,
                                      num_filters=self.encoder.conv5_5.shape.as_list()[3], kernel_size=(1, 1),
                                      l2_strength=self.encoder.wd)
            self._debug(self.expand11)
            self.upscale1 = conv2d_transpose('upscale1', x=self.expand11,is_training= self.is_training,
                                             output_shape=self.encoder.conv5_5.shape.as_list(), batchnorm_enabled=True,
                                             kernel_size=(4, 4), stride=(2, 2), l2_strength=self.encoder.wd)
            self._debug(self.upscale1)
            self.add1 = tf.add(self.upscale1, self.encoder.conv5_5)
            self._debug(self.add1)
            self.expand12 = conv2d('expand1_2', x=self.add1, batchnorm_enabled=True,is_training= self.is_training,
                                      num_filters=self.encoder.conv5_5.shape.as_list()[3], kernel_size=(1, 1),
                                      l2_strength=self.encoder.wd)
            self._debug(self.expand12)

        with tf.name_scope('upscale_2'):
            self.expand21 = conv2d('expand2_1', x=self.expand12, batchnorm_enabled=True,is_training= self.is_training,
                                      num_filters=self.encoder.conv4_1.shape.as_list()[3], kernel_size=(1, 1),
                                      l2_strength=self.encoder.wd)
            self._debug(self.expand21)
            self.upscale2 = conv2d_transpose('upscale2', x=self.expand21,is_training= self.is_training,
                                             output_shape=self.encoder.conv4_1.shape.as_list(),batchnorm_enabled=True,
                                             kernel_size=(4, 4), stride=(2, 2), l2_strength=self.encoder.wd)
            self._debug(self.upscale2)
            self.add2 = tf.add(self.upscale2, self.encoder.conv4_1)
            self._debug(self.add2)
            self.expand22 = conv2d('expand2_2', x=self.add2, batchnorm_enabled=True,is_training= self.is_training,
                              num_filters=self.encoder.conv4_1.shape.as_list()[3], kernel_size=(1, 1),
                              l2_strength=self.encoder.wd)
            self._debug(self.expand22)

        with tf.name_scope('upscale_3'):
            self.expand31 = conv2d('expand3_1', x=self.expand22, batchnorm_enabled=True,is_training= self.is_training,
                                      num_filters=self.encoder.conv3_1.shape.as_list()[3], kernel_size=(1, 1),
                                      l2_strength=self.encoder.wd)
            self._debug(self.expand31)
            self.upscale3 = conv2d_transpose('upscale3', x=self.expand31, batchnorm_enabled=True,is_training= self.is_training,
                                             output_shape=self.encoder.conv3_1.shape.as_list(),
                                             kernel_size=(4, 4), stride=(2, 2), l2_strength=self.encoder.wd)
            self._debug(self.upscale3)
            self.add3 = tf.add(self.upscale3, self.encoder.conv3_1)
            self._debug(self.add3)
            self.expand32 = conv2d('expand3_2', x=self.add3, batchnorm_enabled=True,is_training= self.is_training,
                                num_filters=self.encoder.conv3_1.shape.as_list()[3], kernel_size=(1, 1),
                                l2_strength=self.encoder.wd)
            self._debug(self.expand32)

        with tf.name_scope('upscale_4'):
            self.expand41 = conv2d('expand4_1', x=self.expand32, batchnorm_enabled=True,is_training= self.is_training,
                                      num_filters=self.encoder.conv2_1.shape.as_list()[3], kernel_size=(1, 1),
                                      l2_strength=self.encoder.wd)
            self._debug(self.expand41)
            self.upscale4 = conv2d_transpose('upscale4', x=self.expand41, batchnorm_enabled=True,is_training= self.is_training,
                                             output_shape=self.encoder.conv2_1.shape.as_list(),
                                             kernel_size=(4, 4), stride=(2, 2), l2_strength=self.encoder.wd)
            self._debug(self.upscale4)
            self.add4 = tf.add(self.upscale4, self.encoder.conv2_1)
            self._debug(self.add4)
            self.expand42 = conv2d('expand4_2', x=self.add4, batchnorm_enabled=True,is_training= self.is_training,
                                      num_filters=self.encoder.conv2_1.shape.as_list()[3], kernel_size=(1, 1),
                                      l2_strength=self.encoder.wd)
            self._debug(self.expand42)

        with tf.name_scope('upscale_5'):
            self.upscale5 = conv2d_transpose('upscale5', x=self.expand42, batchnorm_enabled=True,is_training= self.is_training,
                                             output_shape=self.x_pl.shape.as_list()[0:3] + [
                                                 self.encoder.conv2_1.shape.as_list()[3]],
                                             kernel_size=(4, 4), stride=(2, 2), l2_strength=self.encoder.wd)
            self._debug(self.upscale5)
            self.expand5 = conv2d('expand5', x=self.upscale5, batchnorm_enabled=True,is_training= self.is_training,
                                      num_filters=self.encoder.conv1_1.shape.as_list()[3], kernel_size=(1, 1),dropout_keep_prob=0.5,
                                      l2_strength=self.encoder.wd)
            self._debug(self.expand5)

        with tf.name_scope('final_score'):
            self.fscore = conv2d('fscore', x=self.expand5,
                                      num_filters=self.params.num_classes, kernel_size=(1, 1),
                                      l2_strength=self.encoder.wd)
            self._debug(self.fscore)

        self.logits = self.fscore
