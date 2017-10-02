import tensorflow as tf
from layers.convolution import shufflenet_unit, conv2d, max_pool_2d, avg_pool_2d
from layers.dense import dense, flatten


class ShuffleNet:
    """ShuffleNet is implemented here!"""

    def __init__(self, x_input, num_classes, train_flag, batchnorm_enabled=True, num_groups=3, weight_decay=4e-5,
                 bias=0.0):
        self.x_input = x_input
        self.train_flag = train_flag
        self.num_classes = num_classes
        self.num_groups = num_groups
        self.bias = bias
        self.wd = weight_decay
        self.batchnorm_enabled = batchnorm_enabled
        self.score_fr = None

        # These feed layers are for the decoder
        self.feed1 = None
        self.feed2 = None

        # A number stands for the num_groups
        # Output channels for conv1 layer
        self.output_channels = {'1': [144, 288, 576], '2': [200, 400, 800], '3': [240, 480, 960], '4': [272, 544, 1088],
                                '8': [384, 768, 1536], 'conv1': 24}

    def __stage(self, x, stage=2, repeat=3):
        if 2 <= stage <= 4:
            stage_layer = shufflenet_unit('stage' + str(stage) + '_0', x=x, w=None, num_groups=self.num_groups,
                                          group_conv_bottleneck=not (stage == 2),
                                          num_filters=self.output_channels[str(self.num_groups)][stage - 2],
                                          stride=(2, 2),
                                          fusion='concat', l2_strength=self.wd, bias=self.bias,
                                          batchnorm_enabled=self.batchnorm_enabled,
                                          is_training=self.train_flag)
            for i in range(1, repeat + 1):
                stage_layer = shufflenet_unit('stage' + str(stage) + '_' + str(i), x=stage_layer, w=None,
                                              num_groups=self.num_groups,
                                              group_conv_bottleneck=True,
                                              num_filters=self.output_channels[str(self.num_groups)][stage - 2],
                                              stride=(1, 1),
                                              fusion='add', l2_strength=self.wd,
                                              bias=self.bias,
                                              batchnorm_enabled=self.batchnorm_enabled,
                                              is_training=self.train_flag)
            return stage_layer
        else:
            raise ValueError("Stage should be from 2 -> 4")

    def build(self):
        print("Building the ShuffleNet..")
        with tf.variable_scope('shufflenet_encoder'):
            conv1 = conv2d('conv1', x=self.x_input, w=None, num_filters=self.output_channels['conv1'],
                           kernel_size=(3, 3),
                           stride=(2, 2), l2_strength=self.wd, bias=self.bias,
                           batchnorm_enabled=self.batchnorm_enabled, is_training=self.train_flag)
            conv1_padded = tf.pad(conv1, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
            max_pool = max_pool_2d(conv1_padded, size=(3, 3), stride=(2, 2), name='max_pool')
            stage2 = self.__stage(max_pool, stage=2, repeat=3)
            stage3 = self.__stage(stage2, stage=3, repeat=7)
            stage4 = self.__stage(stage3, stage=4, repeat=3)

            self.feed1 = stage3
            self.feed2 = stage2

            # First Experiment is to use the group convolution
            self.score_fr = shufflenet_unit('conv_1c_1x1', stage4, None, self.num_groups, num_filters=self.num_classes,
                                            l2_strength=self.wd, bias=self.bias,
                                            is_training=self.train_flag)
            # Second Experiment is to use the regular conv2d
            # self.score_fr = conv2d('conv_1c_1x1', stage4, num_filters=self.num_classes, l2_strength=self.weight_decay,
            #                        kernel_size=(1, 1), batchnorm_enabled=self.batchnorm_enabled)

            print("\nEncoder ShuffleNet is built successfully\n\n")
