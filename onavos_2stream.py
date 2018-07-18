import tensorflow as tf
from models.model_2stream import Onavos_2stream
from configs.onavos_train import OnavosConfigTrain
from utils.weights_utils import dump_weights
from train.trainer import Trainer

def main(_):
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)
    config= tf.ConfigProto()
    config.gpu_options.allow_growth= True
    sess = tf.Session(config=config)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    model = Onavos_2stream(sess, OnavosConfigTrain())
#    model.load_weights('./pascal.pkl')

    trainer=Trainer(OnavosConfigTrain(),model,global_step,sess)
    trainer.train()


if __name__ == '__main__':
    tf.app.run(main)
