import pickle
import tensorflow as tf


def dump_weights(sess):
    all_vars = {}
    for var in tf.all_variables():
        print(var.name, var.shape)
        all_vars[var.name] = var.eval(sess)

    with open('./checkpoints/weights.pkl', 'wb') as f:
        pickle.dump(all_vars, f, pickle.HIGHEST_PROTOCOL)

