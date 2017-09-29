import time
import pickle
import tensorflow as tf


def timeit(f):
    """ Decorator to time Any Function """

    def timed(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()

        print("   [-] %s : %2.5f sec" % (f.__name__, end_time - start_time))
        return result

    return timed


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


def calculate_flops():
    # Print to stdout an analysis of the number of floating point operations in the
    # model broken down by individual operations.
    tf.profiler.profile(
        tf.get_default_graph(),
        options=tf.profiler.ProfileOptionBuilder.float_operation(), cmd='scope')
