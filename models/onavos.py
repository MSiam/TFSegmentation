from models.basic.basic_model import BasicModel
import tensorflow as tf



class Onavos(BasicModel):

    def __init__(self):
        pass
    def build(self):
        pass

    def init_input(self):
        self.img_placeholder = tf.placeholder(tf.float32, shape=(None, None, 3), name="img_placeholder")
        self.label_placeholder = tf.placeholder(tf.uint8, shape=(None, None, 1), name="label_placeholder")
        # self.tag_placeholder = tf.placeholder(tf.string, shape=(), name="tag_placeholder")

