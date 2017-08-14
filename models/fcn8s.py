from models.basic.basic_model import BasicModel
from encoders.VGG import VGG16

class FCN8s(BasicModel):
    """
    FCN8s Model Architecture
    """

    def __init__(self, args):
        super().__init__(args)

    def build(self, pretrained_path= None, num_classes= 20):
        self.num_classes= num_classes
        self.init_input()
        self.init_network(pretrained_path)

    def init_network(self, pretrained_path):
        """
        Building the Network here
        :return:
        """
        encoder= VGG16(pretrained_path, num_classes= self.num_classes)
        encoder.build(self.img_pl)

        #Build Decoding part
        self.upscore2= conv2d_transpose('upscore2', encoder.score_fr, tf.shape(encoder.feed1),
                kernel_size=(4, 4), stride=(2,2), l2_strength= encoder.wd)
        self.score_feed1= conv2d_f('score_feed1', encoder.feed1, self.num_classes, kernel_size= (1,1),
                l2_strength=encoder.wd)
        self.fuse_feed1 = tf.add(self.score_feed1, self.upscore2)

        self.upscore4= conv2d_transpose('upscore4', self.fuse_feed1, tf.shape(encoder.feed2),
                kernel_size=(4, 4), stride=(2,2), l2_strength= encoder.wd)
        self.score_feed2= conv2d_f('score_feed2', encoder.feed2, self.num_classes, kernel_size= (1,1),
                l2_strength=encoder.wd)
        self.fuse_feed2 = tf.add(self.score_feed2, self.upscore4)

        self.upscore8= conv2d_transpose('upscore8', self.fuse_feed2, tf.shape(self.input),
                kernel_size=(16, 16), stride=(8,8), l2_strength= encoder.wd)

        self.logits = tf.reshape(self.upscore8, (-1, self.num_classes))
        self.softmax = tf.nn.softmax(logits)
