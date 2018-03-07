import time
import pickle
import numpy as np
import tensorflow as tf

def get_vars_underscope(scope, name):
    returned_vars= []
    for v in tf.global_variables():
        if scope+'/'+name in v.op.name:
            returned_vars+= [v]
    return returned_vars

def timeit(f):
    """ Decorator to time Any Function """

    def timed(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        seconds = end_time - start_time
        print("   [-] %s : %2.5f sec, which is %2.5f mins, which is %2.5f hours" %
              (f.__name__, seconds, seconds / 60, seconds / 3600))
        return result

    return timed


def _debug(operation):
    print("Layer_name: " + operation.op.name + " -Output_Shape: " + str(operation.shape.as_list()))


def output_confusion_matrix(confusion_matrix, file_name, num_classes):
    file_output = open(file_name, 'w')
    ans = ""
    for i in range(num_classes):
        ans += '{:>10} '.format(str(i))
    file_output.write(ans)
    file_output.write('\n')

    for i in range(num_classes):
        ans = ""
        for j in range(num_classes + 1):
            if j == 0:
                ans += str(i)
            else:
                ans += '{:>10} '.format(str(confusion_matrix[i][j - 1]))
        file_output.write(ans)
        file_output.write('\n')

    file_output.close()


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


@timeit
def get_class_weights(nclasses, npy_file):
    """
    This function get the weights of every class from labels to use it in the loss while training
    :param nclasses: Number of classes of labels
    :param npy_file: the numpy file of the training ex: Y_train.npy
    :return: class_weights: which is a numpy array contain the weights of all classes
    """
    yy = np.load(npy_file)
    label_to_frequency = {}
    for c in range(nclasses):
        class_mask = np.equal(yy, c)
        class_mask = class_mask.astype(np.float32)
        label_to_frequency[c] = np.sum(class_mask)

    # perform the weighing function label-wise and append the label's class weights to class_weights
    class_weights = []
    total_frequency = sum(label_to_frequency.values())
    for label, frequency in label_to_frequency.items():
        class_weight = 1 / np.log(1.02 + (frequency / total_frequency))
        class_weights.append(class_weight)

    class_weights[-1] = 0
    return class_weights


def calculate_flops():
    # Print to stdout an analysis of the number of floating point operations in the
    # model broken down by individual operations.
    tf.profiler.profile(
        tf.get_default_graph(),
        options=tf.profiler.ProfileOptionBuilder.float_operation(), cmd='scope')


def show_parameters():
    tf.profiler.profile(
        tf.get_default_graph(),
        options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter(), cmd='scope')
