import tensorflow as tf
import numpy as np
import pickle
from utils import Constants
from utils.Measures import create_confusion_matrix
from utils import Measures
import glob
from tensorflow.python.training import moving_averages
from collections import namedtuple
from datasets.Loader import load_dataset
from scipy.ndimage import imread
import os
import numpy
from tqdm import tqdm
from utils.one_shot_utils import adjust_results_to_targets, process_forward_result, flip_if_necessary, average_measures


class Onavos():

    def __init__(self, sess, config):
        self.config = config
        self.sess = sess

