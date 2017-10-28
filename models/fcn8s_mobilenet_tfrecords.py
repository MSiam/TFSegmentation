from models.basic.basic_model import BasicModel
from models.encoders.mobilenet import MobileNet
from layers.convolution import conv2d_transpose, conv2d

import tensorflow as tf


class FCN8sMobileNetTFRecords(BasicModel):
    """
    FCN8s with MobileNet as an encoder Model Architecture
    """

    def __init__(self, args):
        super().__init__(args)
        # init encoder
        self.encoder = None
        # init network layers
        self.upscore2 = None
        self.score_feed1 = None
        self.fuse_feed1 = None
        self.upscore4 = None
        self.score_feed2 = None
        self.fuse_feed2 = None
        self.upscore8 = None
        # init tfrecords needs
        self.handle = None
        self.training_iterator = None
        self.validation_iterator = None
        self.next_img = None
        self.training_handle = None
        self.validation_handle = None
        # get the default session
        self.sess = tf.get_default_session()

    def build(self):
        print("\nBuilding the MODEL...")
        self.init_input()
        self.init_tfrecord_input()
        self.init_network()
        self.init_output()
        self.init_train()
        self.init_summaries()
        print("The Model is built successfully\n")

    def init_tfrecord_input(self):
        if self.args.mode == 'train':
            print("USING TF RECORDS")

            # Use `tf.parse_single_example()` to extract data from a `tf.Example`
            # protocol buffer, and perform any additional per-record preprocessing.
            def parser(record):
                keys_to_features = {
                    'height': tf.FixedLenFeature([], tf.int64),
                    'width': tf.FixedLenFeature([], tf.int64),
                    'image_raw': tf.FixedLenFeature([], tf.string),
                    'mask_raw': tf.FixedLenFeature([], tf.string)
                }
                parsed = tf.parse_single_example(record, keys_to_features)

                image = tf.cast(tf.decode_raw(parsed['image_raw'], tf.uint8), tf.float32)
                annotation = tf.cast(tf.decode_raw(parsed['mask_raw'], tf.uint8), tf.int32)

                height = tf.cast(parsed['height'], tf.int32)
                width = tf.cast(parsed['width'], tf.int32)

                image_shape = tf.stack([height, width, 3])

                annotation_shape = tf.stack([height, width])

                image = tf.reshape(image, image_shape)
                annotation = tf.reshape(annotation, annotation_shape)

                return image, annotation

            # Use `Dataset.map()` to build a pair of a feature dictionary and a label
            # tensor for each example.
            train_filename = "./data/" + self.args.tfrecord_train_file
            train_dataset = tf.contrib.data.TFRecordDataset(train_filename)
            train_dataset = train_dataset.map(parser)
            train_dataset = train_dataset.shuffle(buffer_size=self.args.tfrecord_train_len)
            train_dataset = train_dataset.batch(self.args.batch_size)
            train_dataset = train_dataset.repeat()

            val_filename = "./data/" + self.args.tfrecord_val_file
            val_dataset = tf.contrib.data.TFRecordDataset(val_filename)
            val_dataset = val_dataset.map(parser)
            val_dataset = val_dataset.batch(self.args.batch_size)

            self.training_iterator = train_dataset.make_one_shot_iterator()
            self.validation_iterator = val_dataset.make_initializable_iterator()

            self.training_handle = self.sess.run(self.training_iterator.string_handle())
            self.validation_handle = self.sess.run(self.validation_iterator.string_handle())

            self.handle = tf.placeholder(tf.string, shape=[])
            iterator = tf.contrib.data.Iterator.from_string_handle(self.handle,
                                                                   train_dataset.output_types,
                                                                   train_dataset.output_shapes)

            self.next_img = iterator.get_next()
            self.x_pl, self.y_pl = self.next_img
            self.x_pl.set_shape([None, self.args.img_height, self.args.img_width, 3])
            self.y_pl.set_shape([None, self.args.img_height, self.args.img_width])

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
        with tf.name_scope('upscore_2s'):
            self.upscore2 = conv2d_transpose('upscore2', x=self.encoder.score_fr,
                                             output_shape=[self.args.batch_size] +
                                                          self.encoder.feed1.shape.as_list()[1:3] +
                                                          [self.params.num_classes],
                                             kernel_size=(4, 4), stride=(2, 2), l2_strength=self.encoder.wd)
            self.score_feed1 = conv2d('score_feed1', x=self.encoder.feed1,
                                      num_filters=self.params.num_classes, kernel_size=(1, 1),
                                      l2_strength=self.encoder.wd)
            self.fuse_feed1 = tf.add(self.score_feed1, self.upscore2)

        with tf.name_scope('upscore_4s'):
            self.upscore4 = conv2d_transpose('upscore4', x=self.fuse_feed1,
                                             output_shape=[self.args.batch_size] +
                                                          self.encoder.feed2.shape.as_list()[1:3] +
                                                          [self.params.num_classes],
                                             kernel_size=(4, 4), stride=(2, 2), l2_strength=self.encoder.wd)
            self.score_feed2 = conv2d('score_feed2', x=self.encoder.feed2,
                                      num_filters=self.params.num_classes, kernel_size=(1, 1),
                                      l2_strength=self.encoder.wd)
            self.fuse_feed2 = tf.add(self.score_feed2, self.upscore4)

        with tf.name_scope('upscore_8s'):
            self.upscore8 = conv2d_transpose('upscore8', x=self.fuse_feed2,
                                             output_shape=[self.args.batch_size] + self.x_pl.shape.as_list()[1:3] +
                                                          [self.params.num_classes],
                                             kernel_size=(16, 16), stride=(8, 8), l2_strength=self.encoder.wd)

        self.logits = self.upscore8
