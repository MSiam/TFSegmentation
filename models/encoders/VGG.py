from layers.convolution import conv2d_pre

class VGG16(object):
    '''
    VGG 16 Encoder class
    '''

    VGG_MEAN = [103.939, 116.779, 123.68]

    def __init__(self, pretrained_path, reduced= False):
        '''
        reduced: whether to use vgg reduced with 512 channel for last two layers, or full with 4096 channels
        pretrained_path: path to pretrained weights for vgg16 in npy format
        '''
        # Load pretrained path
        self.pretrained_weights = np.load(pretrained_path, encoding='latin1').item()
        print('pretrained weights loaded')

        self.reduced_flag= reduced
        self.wd = 5e-4

    def build(self, img_input, train=False, num_classes=20):
        """
        Build the VGG model using loaded weights
        Parameters
        ----------
        img_input: image batch tensor
            Image in rgb shap. Scaled to Intervall [0, 255]
        train: bool
            Whether to build train or inference graph
        num_classes: int
            How many classes should be predicted (by fc8)
        """
        # Convert RGB to BGR

        with tf.name_scope('Processing'):
            red, green, blue = tf.split(img_input, 3, 3)
            # assert red.get_shape().as_list()[1:] == [224, 224, 1]
            # assert green.get_shape().as_list()[1:] == [224, 224, 1]
            # assert blue.get_shape().as_list()[1:] == [224, 224, 1]
            img_input = tf.concat([
                blue - VGG16.VGG_MEAN[0],
                green - VGG16.VGG_MEAN[1],
                red - VGG16.VGG_MEAN[2],
            ], 3)

        self.conv1_1= load_conv_layer(img_input, 'conv1_1')
        self.conv1_2= load_conv_layer(self.conv1_1, 'conv1_2', pooling=True)

        self.conv2_1= load_conv_layer(self.conv1_2, 'conv2_1')
        self.conv2_2= load_conv_layer(self.conv2_1, 'conv2_2', pooling=True)

        self.conv3_1= load_conv_layer(self.conv2_2, 'conv3_1')
        self.conv3_2= load_conv_layer(self.conv3_1, 'conv3_2')
        self.conv3_3= load_conv_layer(self.conv3_2, 'conv3_3', pooling=True)

        self.conv4_1= load_conv_layer(self.conv3_3, 'conv4_1')
        self.conv4_2= load_conv_layer(self.conv4_1, 'conv4_2')
        self.conv4_3= load_conv_layer(self.conv4_2, 'conv4_3', pooling=True)

        self.conv5_1= load_conv_layer(self.conv4_3, 'conv5_1')
        self.conv5_2= load_conv_layer(self.conv5_1, 'conv5_2')
        self.conv5_3= load_conv_layer(self.conv5_2, 'conv5_3', pooling=True)

        self.fc6= load_fc_layer(self.conv5_3, 'fc6', activation= tf.nn.relu, dropout=0.5, train=train)
        self.fc7= load_fc_layer(self.fc6, 'fc7', activation= tf.nn.relu, dropout=0.5, train=train)
        self.score_fr= load_fc_layer(self.fc7, 'score_fr', num_classes= num_classes)


    def load_fc_layer(self, bottom, name, num_classes=20,
            activation= None, dropout= 1.0, train= False):
        '''
        Load fully connected layers from pretrained weights in case of full vgg
        in case of reduced vgg initialize randomly
        '''
        if not self.reduced:
            if name == 'fc6':
                w = self.get_fc_weight_reshape(name, [7, 7, 512, 4096])
            elif name == 'score_fr':
                name= 'fc8'
                w = self.get_fc_weight_reshape(name, [1, 1, 4096, 1000],
                                                  num_classes=num_classes)
            else:
                w = self.get_fc_weight_reshape(name, [1, 1, 4096, 4096])

            biases = self.get_bias(name, num_classes=num_classes)
            return conv2d_f_pre(name, bottom, w, l2_strength= self.wd, bias=biases,
                    activation=activation, dropout_keep_prob= dropout, is_training= train)
        else:
            if name == 'fc6':
                num_channels= 512
                kernel_size= (7,7)
            elif name == 'score_fr':
                name= 'fc8'
                num_channels= num_classes
                kernel_size= (1,1)
            else:
                num_channels= 512
                kernel_size= (1,1)

            return conv2d_f(name, bottom, num_channel, kernel_size,= kernel_size, l2_strength=self.wd,
                   activation=activation, dropout_keep_prob=dropout,is_training=train)


    def load_conv_layer(self,bottom, name, pooling= False):
        w = self.get_conv_filter(name)
        biases = self.get_bias(name)
        return conv2d_f_pre(name, bottom, w, l2_strength= self.wd, bias=biases,
                activation=tf.nn.relu, max_pool_enabled=pooling)

    '''
    Next Functions are helpers for loading pretrained weights
    '''
    def get_conv_filter(self, name):
        init = tf.constant_initializer(value=self.pretrained_weights[name][0],
                                       dtype=tf.float32)
        shape = self.pretrained_weights[name][0].shape
        print('Layer name: %s' % name)
        print('Layer shape: %s' % str(shape))
        var = tf.get_variable(name="filter", initializer=init, shape=shape)
        if not tf.get_variable_scope().reuse:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd,
                                       name='weight_loss')
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                 weight_decay)
        return var

    def get_bias(self, name, num_classes=None):
        bias_wights = self.pretrained_weights[name][1]
        shape = self.pretrained_weights[name][1].shape
        if name == 'fc8':
            bias_wights = self._bias_reshape(bias_wights, shape[0],
                                             num_classes)
            shape = [num_classes]
        init = tf.constant_initializer(value=bias_wights,
                                       dtype=tf.float32)
        var = tf.get_variable(name="biases", initializer=init, shape=shape)
        return var

    def get_fc_weight_reshape(self, name, shape, num_classes=None):
        print('Layer name: %s' % name)
        print('Layer shape: %s' % shape)
        weights = self.pretrained_weights[name][0]
        weights = weights.reshape(shape)
        if num_classes is not None:
            weights = self._summary_reshape(weights, shape,
                                            num_new=num_classes)
        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        var = tf.get_variable(name="weights", initializer=init, shape=shape)
        return var


