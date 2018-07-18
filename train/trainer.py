import tensorflow as tf

import utils.Constants
# from Log import log
from utils.utils import average_gradients
import time
from utils import Measures

PROFILE = False
if PROFILE:
    first_run = True


def get_options():
    global first_run
    if PROFILE and not first_run:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        return run_options, run_metadata
    else:
        return None, None


class Trainer(object):
    def __init__(self, config, model, global_step, session):
        self.config = config
        self.model = model
        self.measures = config.measures
        self.opt_str = config.optimizer
        self.train_network = model.train_net
        self.test_network = model.test_net
        self.sess = session
        self.global_step = global_step
        self.learning_rates = config.learning_rates
        assert 1 in self.learning_rates, "no initial learning rate specified"
        self.curr_learning_rate = self.learning_rates[1]
        self.lr_var = tf.placeholder('float32', shape=[], name="learning_rate")
        self.loss_scale_var = tf.placeholder_with_default(1.0, shape=[], name="loss_scale")
        self.opt, self.reset_opt_op = self.create_optimizer(config)
        if self.train_network is not None:
            self.step_op = self.create_step_op()
        else:
            self.step_op = None
            self.update_ops = None

        self.train_writer= tf.summary.FileWriter('summaries/', self.sess.graph)
        init = tf.global_variables_initializer()
        self.sess.run(init)

        if self.config.load_model:
            self.model.load()

    def create_optimizer(self, config):
        momentum = config.momentum
        if self.opt_str == "sgd_nesterov":
            return tf.train.MomentumOptimizer(self.lr_var, momentum, use_nesterov=True), None
        elif self.opt_str == "sgd_momentum":
            return tf.train.MomentumOptimizer(self.lr_var, momentum), None
        elif self.opt_str == "sgd":
            return tf.train.GradientDescentOptimizer(self.lr_var), None
        elif self.opt_str == "adam":
            opt = tf.train.AdamOptimizer(self.lr_var)
            all_vars = tf.global_variables()
            opt_vars = [v for v in all_vars if "Adam" in v.name]
            reset_opt_op = tf.variables_initializer(opt_vars, "reset_optimizer")
            return opt, reset_opt_op
        else:
            assert False, ("unknown optimizer", self.opt_str)

    def reset_optimizer(self):
        assert self.opt_str == "adam", "reset not implemented for other optimizers yet"
        assert self.reset_opt_op is not None
        self.sess.run(self.reset_opt_op)

    def create_step_op(self):
        losses, regularizer = self.train_network['loss'], self.model.regularizer,
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)

        losses_witsh_regularizers = losses + reg_term
        losses_with_regularizers = losses_witsh_regularizers * self.loss_scale_var
        step_op = self.opt.minimize(losses_with_regularizers, self.global_step)

        # step_op = self.opt.apply_gradients(grads, global_step=self.global_step)
        return step_op

    def validation_step(self, _):
        feed_dict = {self.model.is_training: False}
        ops = [self.train_network['loss'], self.train_network['measures']]
        res = self.sess.run(ops, feed_dict=feed_dict)
        loss_summed, measures_accumulated = res
        return loss_summed, measures_accumulated

    def adjust_learning_rate(self, epoch, learning_rate=None):
        if learning_rate is None:
            key = max([k for k in self.learning_rates.keys() if k <= epoch + 1])
            new_lr = self.learning_rates[key]
        else:
            new_lr = learning_rate
        if self.curr_learning_rate != new_lr:
            print("changing learning rate to", new_lr)
            self.curr_learning_rate = new_lr

    def train_step(self, epoch, feed_dict=None, loss_scale=1.0, learning_rate=None):
        self.adjust_learning_rate(epoch, learning_rate)
        if feed_dict is None:
            feed_dict = {}
        else:
            feed_dict = feed_dict.copy()
        feed_dict[self.lr_var] = self.curr_learning_rate
        feed_dict[self.loss_scale_var] = loss_scale
        feed_dict[self.model.is_training] = not self.config.freeze_batchnorm


        # ops = [self.global_step, self.step_op, self.train_network['loss'],
        #        self.train_network['measures'], self.model.merged_summs]
        ops = [self.global_step, self.step_op, self.train_network['loss'],
               self.train_network['measures']]
        run_options, run_metadata = get_options()
        res = self.sess.run(ops, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
        _, _, loss_summed, measures_accumulated = res
        # self.step_n+= 1
        # self.train_writer.add_summary(merged, self.step_n)

        return loss_summed, measures_accumulated

    @staticmethod
    def run_epoch(step_fn, data, epoch):
        loss_total = 0.0
        n_imgs_per_epoch = data.num_examples_per_epoch()
        measures_accumulated = {}
        n_imgs_processed = 0
        while n_imgs_processed < n_imgs_per_epoch:
            start = time.time()
            loss_summed, measures = step_fn(epoch)
            loss_total += loss_summed
            measures_accumulated = Measures.calc_measures_sum(measures_accumulated, measures)
            n_imgs_processed += 1
            measures_avg = Measures.calc_measures_avg(measures, 1, data.ignore_classes)
            end = time.time()
            elapsed = end - start
            print(n_imgs_processed, '/', n_imgs_per_epoch, loss_summed, measures_avg, "elapsed", elapsed)

        loss_total /= n_imgs_processed
        measures_accumulated = Measures.calc_measures_avg(measures_accumulated, n_imgs_processed, data.ignore_classes)
        return loss_total, measures_accumulated

    def train(self):
        print("starting training")
        f = open(self.config.logfile_path, 'w+')
        self.step_n= 0
        for epoch in range(0, self.config.n_epochs):
            f.write("epoch "+str(epoch)+": \n")

            start = time.time()

            train_loss, train_measures = self.run_epoch(self.train_step, self.model.train_data, epoch)
            f.write("train loss : "+str(train_loss)+" , train iou :"+str(train_measures['iou'])+"\n")

            valid_loss, valid_measures = self.run_epoch(self.validation_step, self.model.valid_data, epoch)
            f.write("valid loss : "+str(valid_loss)+" , valid iou :"+str(valid_measures['iou'])+"\n")
            f.flush()
            end = time.time()
            elapsed = end - start
            train_error_string = Measures.get_error_string(train_measures, "train")
            valid_error_string = Measures.get_error_string(valid_measures, "valid")
            print("epoch", epoch + 1, "finished. elapsed:", "%.5f" % elapsed, "train_score:", "%.5f" % train_loss,
                  train_error_string, "valid_score:", valid_loss, valid_error_string)
            if self.config.save:
                self.model.save()
