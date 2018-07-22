import json
from bunch import Bunch
import os
from easydict import EasyDict


def jsonKeys2int(x):
    if isinstance(x, dict):
        return {modified_int(k): v for k, v in x.items()}
    return x


def modified_int(x):
    try:
        out = int(x)
        return out
    except:
        return x


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file,object_hook=jsonKeys2int)

    # convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)

    return config, config_dict


def process_config(jsonfile):
    config, _ = get_config_from_json(jsonfile)
    config.summary_dir = os.path.join("./experiments", config.exp_name, "summary/")
    config.checkpoint_dir = os.path.join("./experiments", config.exp_name, "checkpoint/")
    return config
