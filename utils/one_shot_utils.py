import tensorflow as tf
from utils import Measures
import numpy as np
def adjust_results_to_targets( y_softmax, targets):
    # scale it up!
    return tf.image.resize_images(y_softmax, tf.shape(targets)[1:3])


def process_forward_result( y_argmax, logit, target, tag):
    measures = Measures.compute_measures_for_binary_segmentation(y_argmax, target)

    return measures
def average_measures( measures_dicts):
    keys = measures_dicts[0].keys()
    averaged_measures = {}
    for k in keys:
        vals = [m[k] for m in measures_dicts]
        val = np.mean(vals)
        averaged_measures[k] = val
    return averaged_measures

def flip_if_necessary( y, index_img):
    assert y.shape[0] == 1
    assert index_img.shape[0] == 1
    if all(index_img[0, 0, 0] == [0, 0]):
        flip = False
    elif all(index_img[0, 0, -1] == [0, 0]):
        flip = True
    else:
        assert False, "unexpected index img, probably unsupported augmentors were used during test time"
    if flip:
        return y[:, :, ::-1, :]
    else:
        return y
