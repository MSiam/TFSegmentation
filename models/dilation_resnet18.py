from models.basic.basic_model import BasicModel
from models.encoders.VGG import VGG16
from models.encoders.resnet_18 import RESNET18
from layers.convolution import conv2d_transpose, conv2d

import tensorflow as tf
from utils.misc import _debug

class DilationResNet18(BasicModel):
    """
    FCN8s with MobileNet as an encoder Model Architecture
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

        # Init MobileNet as an encoder
        self.encoder = RESNET18(x_input=self.x_pl,
                                num_classes=self.params.num_classes,
                                pretrained_path=self.args.pretrained_path,
                                train_flag=self.is_training,
                                weight_decay=self.args.weight_decay)

        # Build Encoding part
        self.encoder.build()

        # Build Decoding part
        with tf.name_scope('dilation_2'):
            with tf.variable_scope('conv4_x_dil'):
                self.conv4 = self.encoder._residual_block('conv4_1_dil', self.encoder.conv3, 256, pool_first=False, strides=1, dilation= 2)
                _debug(self.conv4)
                self.conv4 = self.encoder._residual_block('conv4_2_dil', self.conv4, 256)
                _debug(self.conv4)

            with tf.variable_scope('conv5_x_dil'):
                self.conv5 = self.encoder._residual_block('conv5_1_dil', self.conv4, 512, pool_first=False, strides=1, dilation=4)
                _debug(self.conv5)
                self.conv5 = self.encoder._residual_block('conv5_2_dil', self.conv5, 512)
                _debug(self.conv5)

            self.score_fr = conv2d('score_fr_dil', x=self.conv5, num_filters=self.params.num_classes,
                                        kernel_size=(1, 1), l2_strength=self.encoder.wd,
                                        is_training=self.is_training )
            _debug(self.score_fr)

            self.upscore8 = conv2d_transpose('upscore8', x=self.score_fr,
                                             output_shape=self.x_pl.shape.as_list()[0:3] + [self.params.num_classes],
                                             kernel_size=(16, 16), stride=(8, 8), l2_strength=self.encoder.wd, is_training= self.is_training)
            _debug(self.upscore8)

        self.logits= self.upscore8

