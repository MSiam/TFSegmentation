import tensorflow as tf
from layers.convolution import depthwise_separable_conv2d, conv2d
import os
from utils.misc import load_obj, save_obj, _debug


class MobileNet:
    """
    MobileNet Class
    """

    #    MEAN = [103.939, 116.779, 123.68]
    MEAN = [73.29132098, 83.04442645, 72.5238962]

    def __init__(self, x_input,
                 num_classes,
                 pretrained_path,
                 train_flag,
                 width_multipler=1.0,
                 weight_decay=5e-4):

        # init parameters and input
        self.x_input = x_input
        self.num_classes = num_classes
        self.train_flag = train_flag
        self.wd = weight_decay
        self.pretrained_path = os.path.realpath(os.getcwd()) + "/" + pretrained_path
        self.width_multiplier = width_multipler

        # All layers
        self.conv1_1 = None

        self.conv2_1 = None
        self.conv2_2 = None

        self.conv3_1 = None
        self.conv3_2 = None

        self.conv4_1 = None
        self.conv4_2 = None

        self.conv5_1 = None
        self.conv5_2 = None
        self.conv5_3 = None
        self.conv5_4 = None
        self.conv5_5 = None
        self.conv5_6 = None

        self.conv6_1 = None
        self.flattened = None

        self.score_fr = None

        # These feed layers are for the decoder
        self.feed1 = None
        self.feed2 = None

    def build(self):
        self.encoder_build()

    def encoder_build(self):
        print("Building the MobileNet..")
        with tf.variable_scope('mobilenet_encoder'):
            with tf.name_scope('Pre_Processing'):
                red, green, blue = tf.split(self.x_input, num_or_size_splits=3, axis=3)
                preprocessed_input = tf.concat([
                    tf.subtract(blue, MobileNet.MEAN[0]) / tf.constant(255.0),
                    tf.subtract(green, MobileNet.MEAN[1]) / tf.constant(255.0),
                    tf.subtract(red, MobileNet.MEAN[2]) / tf.constant(255.0),
                ], 3)

            self.conv1_1 = conv2d('conv_1', preprocessed_input, num_filters=int(round(32 * self.width_multiplier)),
                                  kernel_size=(3, 3),
                                  padding='SAME', stride=(2, 2), activation=tf.nn.relu, batchnorm_enabled=True,
                                  is_training=self.train_flag, l2_strength=self.wd)
            _debug(self.conv1_1)
            self.conv2_1 = depthwise_separable_conv2d('conv_ds_2', self.conv1_1, width_multiplier=self.width_multiplier,
                                                      num_filters=64, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd, activation=tf.nn.relu)
            _debug(self.conv2_1)
            self.conv2_2 = depthwise_separable_conv2d('conv_ds_3', self.conv2_1, width_multiplier=self.width_multiplier,
                                                      num_filters=128, kernel_size=(3, 3), padding='SAME',
                                                      stride=(2, 2), activation=tf.nn.relu,
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            _debug(self.conv2_2)
            self.conv3_1 = depthwise_separable_conv2d('conv_ds_4', self.conv2_2, width_multiplier=self.width_multiplier,
                                                      num_filters=128, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1), activation=tf.nn.relu,
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            _debug(self.conv3_1)
            self.conv3_2 = depthwise_separable_conv2d('conv_ds_5', self.conv3_1, width_multiplier=self.width_multiplier,
                                                      num_filters=256, kernel_size=(3, 3), padding='SAME',
                                                      stride=(2, 2), activation=tf.nn.relu,
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            _debug(self.conv3_2)
            self.conv4_1 = depthwise_separable_conv2d('conv_ds_6', self.conv3_2, width_multiplier=self.width_multiplier,
                                                      num_filters=256, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1), activation=tf.nn.relu,
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            _debug(self.conv4_1)
            self.conv4_2 = depthwise_separable_conv2d('conv_ds_7', self.conv4_1, width_multiplier=self.width_multiplier,
                                                      num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                      stride=(2, 2), activation=tf.nn.relu,
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            _debug(self.conv4_2)
            self.conv5_1 = depthwise_separable_conv2d('conv_ds_8', self.conv4_2, width_multiplier=self.width_multiplier,
                                                      num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1), activation=tf.nn.relu,
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            _debug(self.conv5_1)
            self.conv5_2 = depthwise_separable_conv2d('conv_ds_9', self.conv5_1, width_multiplier=self.width_multiplier,
                                                      num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1), activation=tf.nn.relu,
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            _debug(self.conv5_2)
            self.conv5_3 = depthwise_separable_conv2d('conv_ds_10', self.conv5_2,
                                                      width_multiplier=self.width_multiplier,
                                                      num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1), activation=tf.nn.relu,
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            _debug(self.conv5_3)
            self.conv5_4 = depthwise_separable_conv2d('conv_ds_11', self.conv5_3,
                                                      width_multiplier=self.width_multiplier,
                                                      num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1), activation=tf.nn.relu,
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            _debug(self.conv5_4)
            self.conv5_5 = depthwise_separable_conv2d('conv_ds_12', self.conv5_4,
                                                      width_multiplier=self.width_multiplier,
                                                      num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1), activation=tf.nn.relu,
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            _debug(self.conv5_5)
            self.conv5_6 = depthwise_separable_conv2d('conv_ds_13', self.conv5_5,
                                                      width_multiplier=self.width_multiplier,
                                                      num_filters=1024, kernel_size=(3, 3), padding='SAME',
                                                      stride=(2, 2), activation=tf.nn.relu,
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            _debug(self.conv5_6)
            self.conv6_1 = depthwise_separable_conv2d('conv_ds_14', self.conv5_6,
                                                      width_multiplier=self.width_multiplier,
                                                      num_filters=1024, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1), activation=tf.nn.relu,
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            _debug(self.conv6_1)
            # Pooling is removed.
            self.score_fr = conv2d('conv_1c_1x1', self.conv6_1, num_filters=self.num_classes, l2_strength=self.wd,
                                   kernel_size=(1, 1))

            _debug(self.score_fr)
            self.feed1 = self.conv4_2
            self.feed2 = self.conv3_2

            print("\nEncoder MobileNet is built successfully\n\n")

    def __restore(self, file_name, sess):
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="mobilenet_encoder")
        try:
            print("Loading ImageNet pretrained weights...")
            dict = load_obj(file_name)
            run_list = []
            for variable in variables:
                for key, value in dict.items():
                    if key in variable.name:
                        run_list.append(tf.assign(variable, value))
            sess.run(run_list)
            print("ImageNet Pretrained Weights Loaded Initially\n\n")
        except:
            print("No pretrained ImageNet weights exist. Skipping...\n\n")

    def load_pretrained_weights(self, sess):
        self.__restore(self.pretrained_path, sess)
