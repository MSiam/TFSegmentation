import tensorflow as tf
from models.model import Onavos
from models.model_1stream import Onavos_1stream
from configs.onavos_config import OnavosConfig
from utils.weights_utils import dump_weights
from train.trainer import Trainer

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model', default_value="onavos",
                           docstring=""" the model name should be in ["onavos","onavos_1stream","onavos_2stream"] """)
tf.app.flags.DEFINE_string('config', default_value="train",
                           docstring=""" the config name should be in ["onavos","train"] """)
tf.app.flags.DEFINE_boolean('train', default_value=True, docstring=""" Whether it is a training or testing""")
tf.app.flags.DEFINE_boolean('oneshot_eval', default_value=True,
                            docstring=""" Whether it is a eval on all data ot not """)
tf.app.flags.DEFINE_boolean('load_weights', default_value=False,
                            docstring=""" Whether it is a eval on all data ot not """)

def main(_):
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)
    sess = tf.Session()

    model = Onavos_1stream(sess, OnavosConfig())
    model.one_shot_evaluation()


if __name__ == '__main__':
    tf.app.run(main)
