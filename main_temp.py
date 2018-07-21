import tensorflow as tf
from models.model import Onavos
from models.model_1stream import Onavos_1stream
from models.model_2stream import Onavos_2stream
from configs.onavos_config import OnavosConfig
from configs.onavos_train import OnavosConfigTrain
from utils.weights_utils import dump_weights
from train.trainer import Trainer
from utils.args import get_args
from utils.config import process_config
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model', default_value="onavos_1stream",
                           docstring=""" the model name should be in ["onavos","onavos_1stream","onavos_2stream"] """)
tf.app.flags.DEFINE_string('config', default_value="onavos",
                           docstring=""" the config name should be in ["onavos","train"] """)
tf.app.flags.DEFINE_boolean('mode', default_value="train", docstring="""The experiment mode""")
tf.app.flags.DEFINE_boolean('load_weights', default_value=False,
                            docstring=""" Whether it is a eval on all data ot not """)


def main(_):

    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)



    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)
    gpu_config= tf.ConfigProto()
    gpu_config.gpu_options.allow_growth= True
    sess = tf.Session(config=gpu_config)

    #creating global step counter
    global_step = tf.Variable(0, name='global_step', trainable=False)

    if FLAGS.config == "onavos":
        config = OnavosConfig()
    elif FLAGS.config == "train":
        config = OnavosConfigTrain()
    else:
        raise Exception('Unknown config')
    if FLAGS.model == "onavos":
        model = Onavos(sess, config)
    elif FLAGS.model == "onavos_1stream":
        model = Onavos_1stream(sess, config)
    elif FLAGS.model == "onavos_2stream":
        model = Onavos_2stream(sess, config)

    else:
        raise Exception('Unknown model')

    if FLAGS.parse_onavos_weights:
        model.parse_onavos_weights('./model4.pkl')

    if FLAGS.mode=="train":
        trainer = Trainer(config, model, global_step, sess)
        trainer.train()
    elif FLAGS.mode=="oneshot":
        model.one_shot_evaluation()
    elif FLAGS.mode=="online":
        trainer = Trainer(config, model, global_step, sess)
        model.online_forward(sess, config, model, trainer)

if __name__ == '__main__':
    tf.app.run(main)
