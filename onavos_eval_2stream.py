import tensorflow as tf
from models.model_2stream import Onavos_2stream
from configs.onavos_config import OnavosConfig
from utils.weights_utils import dump_weights
from train.trainer import Trainer

def main(_):
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)
    sess = tf.Session()

    model = Onavos_2stream(sess, OnavosConfig())
    model.one_shot_evaluation()


if __name__ == '__main__':
    tf.app.run(main)
