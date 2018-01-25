"""
The Basic class to train any Model
"""

import tensorflow as tf

from utils.misc import timeit


class BasicTrain(object):
    """
    A Base class for train classes of the models
    Contain all necessary functions for training
    """

    def __init__(self, args, sess, train_model, test_model):
        print("\nTraining is initializing itself\n")

        self.args = args
        self.sess = sess
        self.model = train_model
        self.test_model = test_model

        # shortcut for model params
        self.params = self.test_model.params


        # To initialize all variables
        self.init = None
        self.init_model()

        # Get the description of the graph
        # self.get_all_variables_in_graph()
        # exit(0)

        # Create a saver object
        self.saver = tf.train.Saver(max_to_keep=self.args.max_to_keep,
                                    #keep_checkpoint_every_n_hours=10,
                                    save_relative_paths=True)

        self.saver_best = tf.train.Saver(max_to_keep=1,
                                         save_relative_paths=True)

        # Load from latest checkpoint if found
        if train_model is not None:
            self.load_model(train_model)

    @timeit
    def init_model(self):
        print("Initializing the variables of the model")
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)
        print("Initialization finished")

    def save_model(self):
        """
        Save Model Checkpoint
        :return:
        """
        print("saving a checkpoint")
        self.saver.save(self.sess, self.args.checkpoint_dir, self.model.global_step_tensor)
        print("Saved a checkpoint")

    def save_best_model(self):
        """
        Save BEST Model Checkpoint
        :return:
        """
        print("saving a checkpoint for the best model")
        self.saver_best.save(self.sess, self.args.checkpoint_best_dir, self.model.global_step_tensor)
        print("Saved a checkpoint for the best model")

    def load_best_model(self):
        """
        Load the best model checkpoint
        :return:
        """
        print("loading a checkpoint for BEST ONE")
        latest_checkpoint = tf.train.latest_checkpoint(self.args.checkpoint_best_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver_best.restore(self.sess, latest_checkpoint)
        else:
            print("ERROR NO best checkpoint found")
            exit(-1)
        print("BEST MODEL LOADED..")

    @timeit
    def load_model(self, model):
        """
        Load the latest checkpoint
        :return:
        """

        try:
            # This is for loading the pretrained weights if they can't be loaded during initialization.
            model.encoder.load_pretrained_weights(self.sess)
            print("Pretrained weights of the encoder is loaded")
        except AttributeError:
            pass

        print("Searching for a checkpoint")
        latest_checkpoint = tf.train.latest_checkpoint(self.args.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)
            print("Model loaded from the latest checkpoint\n")
        else:
            print("\n.. No ckpt, SO First time to train :D ..\n")

    def train(self):
        raise NotImplementedError("train function is not implemented in the trainer")

    def finalize(self):
        raise NotImplementedError("finalize function is not implemented in the trainer")

    def get_all_variables_in_graph(self):
        print('################### Variables of the graph')
        from collections import OrderedDict
        var_dict = OrderedDict()
        for var in tf.all_variables():
            if var.op.name not in var_dict.keys() and var.shape != () and "Adam" not in var.op.name:
                print("'" + str(var.op.name) + "': " + str(var.op.name).replace('/', '_') + "# " + str(var.shape))
                key = var.op.name
                # x = var.eval(self.sess)
                var_dict[key] = var.shape
        print('Finished Display all variables')
