"""
This file will contain the validation of params
"""
import utils.logger

from config.config import parse_config


def require_params(args):
    """
    Validate that all needed params are in
    :param args:
    :return: 0:Success -1:failed
    """
    if args.data_dir is not None and args.exp_dir is not None and args.out_dir is not None \
            and args.img_height is not None \
            and args.img_width is not None \
            and args.num_channels is not None \
            and args.num_classes is not None:
        return 0
    else:
        return -1


def get_params():
    """
    Do the whole process of parsing the arguments and validating it.
    :return args: Validated arguments to agent to use it.
    """
    print("\nParsing Arguments..")
    args = parse_config()
    # Check required params
    if require_params(args) == -1:
        print("ERROR some important params is missing Check require_params function")
        exit(-1)
    return args
