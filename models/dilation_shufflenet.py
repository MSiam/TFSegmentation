from models.basic.basic_model import BasicModel
from models.encoders.shufflenet import ShuffleNet
from layers.convolution import conv2d_transpose, conv2d, atrous_conv2d
from utils.misc import _debug
import tensorflow as tf
import pdb

class DilationShuffleNet(BasicModel):
    """
    FCN8s with ShuffleNet as an encoder Model Architecture
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

        with tf.name_scope('dilation_2'):
            self.stage3 = self.encoder.stage(self.encoder.stage2, stage=3, repeat=7, dilation=2)
            _debug(self.stage3)
            self.stage4 = self.encoder.stage(self.stage3, stage=4, repeat=3, dilation=4)
            _debug(self.stage4)

            self.score_fr = conv2d('score_fr_dil', x=self.stage4, num_filters=self.params.num_classes,
                                        kernel_size=(1, 1), l2_strength=self.encoder.wd,
                                        is_training=self.is_training )
            _debug(self.score_fr)

            self.upscore8 = conv2d_transpose('upscore8', x=self.score_fr,
                                             output_shape=self.x_pl.shape.as_list()[0:3] + [self.params.num_classes],
                                             kernel_size=(16, 16), stride=(8, 8), l2_strength=self.encoder.wd, is_training= self.is_training)
            _debug(self.upscore8)

        self.logits = self.upscore8
