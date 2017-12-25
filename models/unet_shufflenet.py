from models.basic.basic_model import BasicModel
from models.encoders.shufflenet import ShuffleNet
from layers.convolution import conv2d_transpose, conv2d

import tensorflow as tf


class UNetShuffleNet(BasicModel):
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

        # Init ShuffleNet as an encoder
        self.encoder = ShuffleNet(x_input=self.x_pl, num_classes=self.params.num_classes,
                                  pretrained_path=self.args.pretrained_path, train_flag=self.is_training,
                                  batchnorm_enabled=self.args.batchnorm_enabled, num_groups=self.args.num_groups,
                                  weight_decay=self.args.weight_decay, bias=self.args.bias)
        # Build Encoding part
        self.encoder.build()

        # Build Decoding part
        with tf.name_scope('upscale_1'):
            self.expand11 = conv2d('expand1_1', x=self.encoder.stage4, batchnorm_enabled=True,
                                   is_training=self.is_training,
                                   num_filters=self.encoder.stage3.shape.as_list()[3], kernel_size=(1, 1),
                                   l2_strength=self.encoder.wd)
            self._debug(self.expand11)
            self.upscale1 = conv2d_transpose('upscale1', x=self.expand11, is_training=self.is_training,
                                             output_shape=self.encoder.stage3.shape.as_list(), batchnorm_enabled=True,
                                             kernel_size=(4, 4), stride=(2, 2), l2_strength=self.encoder.wd)
            self._debug(self.upscale1)
            self.add1 = tf.add(self.upscale1, self.encoder.stage3)
            self._debug(self.add1)
            self.expand12 = conv2d('expand1_2', x=self.add1, batchnorm_enabled=True, is_training=self.is_training,
                                   num_filters=self.encoder.stage3.shape.as_list()[3], kernel_size=(1, 1),
                                   l2_strength=self.encoder.wd)
            self._debug(self.expand12)

        with tf.name_scope('upscale_2'):
            self.expand21 = conv2d('expand2_1', x=self.expand12, batchnorm_enabled=True, is_training=self.is_training,
                                   num_filters=self.encoder.stage2.shape.as_list()[3], kernel_size=(1, 1),
                                   l2_strength=self.encoder.wd)
            self._debug(self.expand21)
            self.upscale2 = conv2d_transpose('upscale2', x=self.expand21, is_training=self.is_training,
                                             output_shape=self.encoder.stage2.shape.as_list(), batchnorm_enabled=True,
                                             kernel_size=(4, 4), stride=(2, 2), l2_strength=self.encoder.wd)
            self._debug(self.upscale2)
            self.add2 = tf.add(self.upscale2, self.encoder.stage2)
            self._debug(self.add2)
            self.expand22 = conv2d('expand2_2', x=self.add2, batchnorm_enabled=True, is_training=self.is_training,
                                   num_filters=self.encoder.stage2.shape.as_list()[3], kernel_size=(1, 1),
                                   l2_strength=self.encoder.wd)
            self._debug(self.expand22)

        with tf.name_scope('upscale_3'):
            self.expand31 = conv2d('expand3_1', x=self.expand22, batchnorm_enabled=True, is_training=self.is_training,
                                   num_filters=self.encoder.max_pool.shape.as_list()[3], kernel_size=(1, 1),
                                   l2_strength=self.encoder.wd)
            self._debug(self.expand31)
            self.upscale3 = conv2d_transpose('upscale3', x=self.expand31, batchnorm_enabled=True,
                                             is_training=self.is_training,
                                             output_shape=[self.encoder.max_pool.shape[0],
                                                           self.encoder.max_pool.shape.as_list()[1] + 1,
                                                           self.encoder.max_pool.shape.as_list()[2] + 1,
                                                           self.encoder.max_pool.shape.as_list()[3]],
                                             kernel_size=(4, 4), stride=(2, 2), l2_strength=self.encoder.wd)
            self._debug(self.upscale3)
            padded = tf.pad(self.encoder.max_pool, [[0, 0], [0, 1], [0, 1], [0, 0]], "CONSTANT")
            self.add3 = tf.add(self.upscale3, padded)
            self._debug(self.add3)
            self.expand32 = conv2d('expand3_2', x=self.add3, batchnorm_enabled=True, is_training=self.is_training,
                                   num_filters=self.encoder.max_pool.shape.as_list()[3], kernel_size=(1, 1),
                                   l2_strength=self.encoder.wd)
            self._debug(self.expand32)

        with tf.name_scope('upscale_4'):
            self.expand41 = conv2d('expand4_1', x=self.expand32, batchnorm_enabled=True, is_training=self.is_training,
                                   num_filters=self.encoder.conv1.shape.as_list()[3], kernel_size=(1, 1),
                                   l2_strength=self.encoder.wd)
            self._debug(self.expand41)
            self.upscale4 = conv2d_transpose('upscale4', x=self.expand41, batchnorm_enabled=True,
                                             is_training=self.is_training,
                                             output_shape=[self.encoder.conv1.shape[0],
                                                           self.encoder.conv1.shape.as_list()[1] + 1,
                                                           self.encoder.conv1.shape.as_list()[2] + 1,
                                                           self.encoder.conv1.shape.as_list()[3]],
                                             kernel_size=(4, 4), stride=(2, 2), l2_strength=self.encoder.wd)
            self._debug(self.upscale4)
            padded2 = tf.pad(self.encoder.conv1, [[0, 0], [0, 1], [0, 1], [0, 0]], "CONSTANT")
            self.add4 = tf.add(self.upscale4, padded2)
            self._debug(self.add4)
            self.expand42 = conv2d('expand4_2', x=self.add4, batchnorm_enabled=True, is_training=self.is_training,
                                   num_filters=self.encoder.conv1.shape.as_list()[3], kernel_size=(1, 1),
                                   l2_strength=self.encoder.wd)
            self._debug(self.expand42)

        with tf.name_scope('upscale_5'):
            self.upscale5 = conv2d_transpose('upscale5', x=self.expand42, batchnorm_enabled=True,
                                             is_training=self.is_training,
                                             output_shape=self.x_pl.shape.as_list()[0:3] + [
                                                 self.encoder.conv1.shape.as_list()[3]],
                                             kernel_size=(4, 4), stride=(2, 2), l2_strength=self.encoder.wd)
            self._debug(self.upscale5)
            self.expand5 = conv2d('expand5', x=self.upscale5, batchnorm_enabled=True, is_training=self.is_training,
                                  num_filters=self.encoder.conv1.shape.as_list()[3], kernel_size=(1, 1),
                                  dropout_keep_prob=0.5,
                                  l2_strength=self.encoder.wd)
            self._debug(self.expand5)

        with tf.name_scope('final_score'):
            self.fscore = conv2d('fscore', x=self.expand5,
                                 num_filters=self.params.num_classes, kernel_size=(1, 1),
                                 l2_strength=self.encoder.wd)
            self._debug(self.fscore)

        self.logits = self.fscore
