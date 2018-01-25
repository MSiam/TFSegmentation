import tensorflow as tf
from layers.convolution import depthwise_separable_conv2d, conv2d
import os
from utils.misc import load_obj, save_obj


class MobileNet:
    """
    MobileNet Class
    """

#    MEAN = [103.939, 116.779, 123.68]
    MEAN = [73.29132098,  83.04442645,  72.5238962]
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

    @staticmethod
    def _debug(operation):
        print("Layer_name: " + operation.op.name + " -Output_Shape: " + str(operation.shape.as_list()))

    def encoder_build(self):
        print("Building the MobileNet..")
        with tf.variable_scope('mobilenet_encoder'):
            with tf.name_scope('Pre_Processing'):
                red, green, blue = tf.split(self.x_input, num_or_size_splits=3, axis=3)
                preprocessed_input = tf.concat([
                    (blue - MobileNet.MEAN[0]) / 255.0,
                    (green - MobileNet.MEAN[1]) / 255.0,
                    (red - MobileNet.MEAN[2]) / 255.0,
                ], 3)

            self.conv1_1 = conv2d('conv_1', preprocessed_input, num_filters=int(round(32 * self.width_multiplier)),
                                  kernel_size=(3, 3),
                                  padding='SAME', stride=(2, 2), activation=tf.nn.relu6, batchnorm_enabled=True,
                                  is_training=self.train_flag, l2_strength=self.wd)
            self._debug(self.conv1_1)
            self.conv2_1 = depthwise_separable_conv2d('conv_ds_2', self.conv1_1, width_multiplier=self.width_multiplier,
                                                      num_filters=64, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd, activation=tf.nn.relu6)
            self._debug(self.conv2_1)
            self.conv2_2 = depthwise_separable_conv2d('conv_ds_3', self.conv2_1, width_multiplier=self.width_multiplier,
                                                      num_filters=128, kernel_size=(3, 3), padding='SAME',
                                                      stride=(2, 2), activation=tf.nn.relu6,
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            self._debug(self.conv2_2)
            self.conv3_1 = depthwise_separable_conv2d('conv_ds_4', self.conv2_2, width_multiplier=self.width_multiplier,
                                                      num_filters=128, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1), activation=tf.nn.relu6,
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            self._debug(self.conv3_1)
            self.conv3_2 = depthwise_separable_conv2d('conv_ds_5', self.conv3_1, width_multiplier=self.width_multiplier,
                                                      num_filters=256, kernel_size=(3, 3), padding='SAME',
                                                      stride=(2, 2), activation=tf.nn.relu6,
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            self._debug(self.conv3_2)
            self.conv4_1 = depthwise_separable_conv2d('conv_ds_6', self.conv3_2, width_multiplier=self.width_multiplier,
                                                      num_filters=256, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1), activation=tf.nn.relu6,
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            self._debug(self.conv4_1)
            self.conv4_2 = depthwise_separable_conv2d('conv_ds_7', self.conv4_1, width_multiplier=self.width_multiplier,
                                                      num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                      stride=(2, 2), activation=tf.nn.relu6,
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            self._debug(self.conv4_2)
            self.conv5_1 = depthwise_separable_conv2d('conv_ds_8', self.conv4_2, width_multiplier=self.width_multiplier,
                                                      num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1), activation=tf.nn.relu6,
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            self._debug(self.conv5_1)
            self.conv5_2 = depthwise_separable_conv2d('conv_ds_9', self.conv5_1, width_multiplier=self.width_multiplier,
                                                      num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1), activation=tf.nn.relu6,
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            self._debug(self.conv5_2)
            self.conv5_3 = depthwise_separable_conv2d('conv_ds_10', self.conv5_2,
                                                      width_multiplier=self.width_multiplier,
                                                      num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1), activation=tf.nn.relu6,
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            self._debug(self.conv5_3)
            self.conv5_4 = depthwise_separable_conv2d('conv_ds_11', self.conv5_3,
                                                      width_multiplier=self.width_multiplier,
                                                      num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1), activation=tf.nn.relu6,
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            self._debug(self.conv5_4)
            self.conv5_5 = depthwise_separable_conv2d('conv_ds_12', self.conv5_4,
                                                      width_multiplier=self.width_multiplier,
                                                      num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1), activation=tf.nn.relu6,
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            self._debug(self.conv5_5)
            self.conv5_6 = depthwise_separable_conv2d('conv_ds_13', self.conv5_5,
                                                      width_multiplier=self.width_multiplier,
                                                      num_filters=1024, kernel_size=(3, 3), padding='SAME',
                                                      stride=(2, 2), activation=tf.nn.relu6,
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            self._debug(self.conv5_6)
            self.conv6_1 = depthwise_separable_conv2d('conv_ds_14', self.conv5_6,
                                                      width_multiplier=self.width_multiplier,
                                                      num_filters=1024, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1), activation=tf.nn.relu6,
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            self._debug(self.conv6_1)
            # Pooling is removed.
            self.score_fr = conv2d('conv_1c_1x1', self.conv6_1, num_filters=self.num_classes, l2_strength=self.wd,
                                   kernel_size=(1, 1))

            self._debug(self.score_fr)
            self.feed1 = self.conv4_2
            self.feed2 = self.conv3_2

            print("\nEncoder MobileNet is built successfully\n\n")

    def __restore(self, file_name, sess):
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="network/mobilenet_encoder")
        dict = load_obj(file_name)
        for variable in variables:
            for key, value in dict.items():
                print('Layer Loaded ', key)
                if key in variable.name:
                    sess.run(tf.assign(variable, value))

    def load_pretrained_weights(self, sess):
        print("Loading ImageNet Pretrained Weights...")
        # self.__convert_graph_names(os.path.realpath(os.getcwd()) + '/pretrained_weights/mobilenet_v1_vanilla.pkl')
        self.__restore(self.pretrained_path, sess)
        print("ImageNet Pretrained Weights Loaded Initially")

    def __convert_graph_names(self, path):
        """
        This function is to convert from the mobilenet original model pretrained weights structure to our
        model pretrained weights structure.
        :param path: (string) path to the original pretrained weights .pkl file
        :return: None
        """
        dict = load_obj(path)
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='mobilenet_encoder')
        dict_output = {}
        # for variable in variables:
        #     print(variable.name)
        # for key, value in dict.items():
        #     print(key)

        for key, value in dict.items():
            for variable in variables:
                for i in range(len(dict)):
                    for j in range(len(variables)):
                        if ((key.find("Conv2d_" + str(i) + "_") != -1 and variable.name.find(
                                        "conv_ds_" + str(j) + "/") != -1) and i + 1 == j):
                            if key.find("depthwise") != -1 and variable.name.find(
                                    "depthwise") != -1 and (key.find("gamma") != -1 and variable.name.find(
                                "gamma") != -1 or key.find("beta") != -1 and variable.name.find(
                                "beta") != -1) or key.find("pointwise") != -1 and variable.name.find(
                                "pointwise") != -1 and (key.find("gamma") != -1 and variable.name.find(
                                "gamma") != -1 or key.find("beta") != -1 and variable.name.find(
                                "beta") != -1) or key.find("pointwise/weights") != -1 and variable.name.find(
                                "pointwise/weights") != -1 or key.find(
                                "depthwise_weights") != -1 and variable.name.find(
                                "depthwise/weights") != -1 or key.find("pointwise/biases") != -1 and variable.name.find(
                                "pointwise/biases") != -1 or key.find("depthwise/biases") != -1 and variable.name.find(
                                "depthwise/biases") != -1 or key.find("1x1/weights") != -1 and variable.name.find(
                                "1x1/weights") != -1 or key.find("1x1/biases") != -1 and variable.name.find(
                                "1x1/biases") != -1:
                                dict_output[variable.name] = value
                        elif key.find(
                                "Conv2d_0/") != -1 and variable.name.find("conv_1/") != -1:
                            if key.find("weights") != -1 and variable.name.find("weights") != -1 or key.find(
                                    "biases") != -1 and variable.name.find(
                                "biases") != -1 or key.find("beta") != -1 and variable.name.find(
                                "beta") != -1 or key.find("gamma") != -1 and variable.name.find(
                                "gamma") != -1:
                                dict_output[variable.name] = value

        save_obj(dict_output, self.pretrained_path)
        print("Pretrained weights converted to the new structure. The filename is mobilenet_v1.pkl.")
