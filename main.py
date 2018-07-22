import tensorflow as tf
from models.model_1stream import Onavos_1stream
from models.model_2stream import Onavos_2stream
from utils.weights_utils import dump_weights
from train.trainer import Trainer
from utils.args import get_args
from utils.config import process_config
from utils.dirs import create_dirs


def main():
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        raise Exception("Missing or Invalid arguments")
    #
    # creating experiment folder
    create_dirs([config.summary_dir, config.checkpoint_dir])

    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    sess = tf.Session(config=gpu_config)

    # creating global step counter
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # model
    try:
        model_class = globals()[config.model_name]
    except:
        raise Exception("Invalid Model Class Name")
    model = model_class(sess, config)

    # parse weights from original onavos
    if config.parse_onavos_weights:
        model.parse_onavos_weights('./model.pkl')

    # mode [train or oneshot or online]
    if config.mode == "train":
        trainer = Trainer(config, model, global_step, sess)
        trainer.train()
    elif config.mode == "oneshot":
        model.one_shot_evaluation()
    elif config.mode == "online":
        trainer = Trainer(config, model, global_step, sess)
        model.online_forward(sess, config, model, trainer)


if __name__ == '__main__':
    main()
