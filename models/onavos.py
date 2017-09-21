from models.basic.basic_model import BasicModel
import tensorflow as tf
import numpy as np

MAX_ADJUSTABLE_CLASSES = 100  # max 100 objects per sequence should be sufficient


class Onavos():
    def __init__(self):
        # super.__init__(args)
        # self.args = args
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=1e-4)
        self.init_input()
        self.build()

    def init_input(self):
        self.img_placeholder = tf.placeholder(tf.float32, shape=(None, 200, 200, 3), name="img_placeholder")
        self.label_placeholder = tf.placeholder(tf.uint8, shape=(None, 200, 200, 1), name="label_placeholder")
        self.tag_placeholder = tf.placeholder(tf.string, shape=(), name="tag_placeholder")
        self.is_training = tf.placeholder(tf.bool)

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
                    res = tf.layers.conv2d(curr, n_features[-1], 1, strides=strides_res, kernel_regularizer=self.regularizer,
                                           use_bias=False, padding='SAME', name='conv0')
                else:
                    res = tf.layers.conv2d(curr, n_features[-1], 1, kernel_regularizer=self.regularizer,
                                           use_bias=False, padding='SAME', name='conv0')

            if dilations is None:
                curr = tf.layers.conv2d(curr, n_features[0], filter_size[0], strides=strides[0],
                                        kernel_regularizer=self.regularizer, use_bias=False, padding='SAME', name='conv1')
            else:
                curr = tf.layers.conv2d(curr, n_features[0], filter_size[0], dilation_rate=dilations[0],
                                        kernel_regularizer=self.regularizer, use_bias=False, padding='SAME', name='conv1')

            for idx in range(1, n_convs):
                curr = tf.layers.batch_normalization(inputs, axis=-1, momentum=batch_norm_decay, epsilon=1e-5,
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

    def segmentation_softmax(self, name, inp, targets, n_classes, void_label, tower_setup, filter_size=(1, 1),
                             input_activation=None, dilation=None, resize_targets=False, resize_logits=False, loss="ce",
                             fraction=None):
        n_features_inp = int(inp.get_shape()[-1])
        filter_size = list(filter_size)

        with tf.variable_scope(name):
            if input_activation is not None:
                inp = input_activation(inp)

            if self.n_classes_current is None:
                self.n_classes_current = n_classes

            if dilation is None:
                y_pred = tf.layers.conv2d(inp, n_classes, filter_size[0], kernel_regularizer=self.regularizer,
                                          padding='SAME', name='conv')
            else:
                y_pred = tf.layers.conv2d(inp, n_classes, filter_size[0], kernel_regularizer=self.regularizer,
                                          dilation_rate=dilation, padding='SAME', name='conv')
            self.outputs = [tf.nn.softmax(y_pred, -1, 'softmax')]

            if resize_targets:
                targets = tf.image.resize_nearest_neighbor(targets, tf.shape(y_pred)[1:3])
            if resize_logits:
                y_pred = tf.image.resize_images(y_pred, tf.shape(targets)[1:3])

            pred = tf.argmax(y_pred, axis=3)
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

            self.loss = self.create_loss(loss, fraction, no_void_label_mask, targets, tower_setup, void_label, y_pred)

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

    def create_loss(self, loss_str, fraction, no_void_label_mask, targets, tower_setup, void_label, y_pred):
        ce = None
        if "ce" in loss_str:
            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=targets, name="ce")

            if void_label is not None:
                mask = tf.cast(no_void_label_mask, tower_setup.dtype)
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

    def build(self):
        conv0 = tf.layers.conv2d(self.img_placeholder, 64, 3, kernel_regularizer=self.regularizer,
                                 use_bias=False, padding='SAME', name='conv0')

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
                                              training=self.is_training, name='con1/bn')
        conv1 = tf.layers.conv2d(conv1, 512, 3, dilation_rate=12, kernel_regularizer=self.regularizer,
                                 padding='SAME', name='conv1')
        conv1 = tf.nn.relu(conv1)

        self.segmentation_softmax('output', conv1, self.label_placeholder, 2, filter_size=(3, 3), input_activation=tf.nn.relu,
                                  dilation=12, resize_targets=True, loss="bootstrapped_ce", )


tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.INFO)

sess = tf.Session()

model = Onavos()

