from models.basic.basic_model import BasicModel
from models.encoders.shufflenet import ShuffleNet
from layers.convolution import conv2d_transpose, conv2d, atrous_conv2d

import tensorflow as tf


class FCN8sShuffleNetUpsample(BasicModel):
    """
    FCN8s with ShuffleNet Upsampling 2x2 as an encoder Model Architecture
    """

    def __init__(self, args, phase=0):
        super().__init__(args, phase=phase)
        # init encoder
        self.encoder = None
        # init network layers

    def build(self):
        print("\nBuilding the MODEL...")
        self.init_input()
        self.init_network()
        self.init_output()
        self.init_train()
        self.init_summaries()
        print("The Model is built successfully\n")

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
        with tf.name_scope('upscore_2s'):
            shape = self.encoder.score_fr.shape.as_list()[1:3]
            upscore2_upsample = tf.image.resize_images(self.encoder.score_fr,(2 * shape[0], 2 * shape[1]))
            self.upscore2 = conv2d('upscore2', x=upscore2_upsample, num_filters=self.params.num_classes,
                                   l2_strength=self.encoder.wd)
            self.score_feed1 = conv2d('score_feed1', x=self.encoder.feed1,
                                      num_filters=self.params.num_classes, kernel_size=(1, 1),
                                      l2_strength=self.encoder.wd)
            self.fuse_feed1 = tf.add(self.score_feed1, self.upscore2)

        with tf.name_scope('upscore_4s'):
            shape = self.fuse_feed1.shape.as_list()[1:3]
            upscore4_upsample = tf.image.resize_images(self.fuse_feed1,(2 * shape[0], 2 * shape[1]))
            self.upscore4 = conv2d('upscore4', x=upscore4_upsample, num_filters=self.params.num_classes,
                                   l2_strength=self.encoder.wd)

            self.score_feed2 = conv2d('score_feed2', x=self.encoder.feed2,
                                      num_filters=self.params.num_classes, kernel_size=(1, 1),
                                      l2_strength=self.encoder.wd)
            self.fuse_feed2 = tf.add(self.score_feed2, self.upscore4)

        with tf.name_scope('upscore_8s'):
            shape = self.fuse_feed2.shape.as_list()[1:3]
            upscore8_upsample = tf.image.resize_images(self.fuse_feed2,(8 * shape[0], 8 * shape[1]))
            self.upscore8 = conv2d('upscore8', x=upscore8_upsample, num_filters=self.params.num_classes,
                                   l2_strength=self.encoder.wd)

        self.logits = self.upscore8