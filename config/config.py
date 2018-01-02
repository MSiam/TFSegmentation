"""
This file will contain the parsing the arguments it and its description
"""
import os
import yaml
import argparse
import utils.logger


def visualize_config(args):
    """
    Visualize the configuration on the terminal to check the state
    :param args:
    :return:
    """
    print("\nUsing this arguments check it\n")
    for key, value in sorted(vars(args).items()):
        if value is not None:
            print("{} -- {} --".format(key, value))
    print("\n\n")


def parse_config():
    """
    Parse Configuration of the run from YAML file
    :return args:
    """

    # Create a parser
    parser = argparse.ArgumentParser(description="Segmentation using tensorflow")
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    # parse the configuration file if given
    parser.add_argument('--load_config', default=None,
                        dest='config_path',
                        help='load config from a yaml file so specify it and put it in ./config/experiments_config/')
    args, unprocessed_args = parser.parse_known_args()
    config_path = args.config_path

    # Modules arguments
    parser.add_argument('mode',
                        choices=('train_n_test', 'train', 'test', 'overfit', 'inference', 'inference_pkl', 'debug'),
                        default=None, help='Mode of operation')
    parser.add_argument('operator', default=None, help='Operator class (trainer of tester)')
    parser.add_argument('model', default=None, help='Model class to operate on')

    # Directories arguments
    parser.add_argument('--data_dir', default=None, help='The data folder')
    parser.add_argument('--abs_data_dir', default=None, help='The data folder')
    parser.add_argument('--tfrecord_train_file', default=None, help='the tf_record to use it ')
    parser.add_argument('--tfrecord_val_file', default=None, help='the tf_record to use it ')
    parser.add_argument('--tfrecord_train_len', default=None, help='the tf_record to use it ')
    parser.add_argument('--tfrecord_val_len', default=None, help='the tf_record to use it ')
    parser.add_argument('--h5_train_file', default=None, help='the h5 to use it ')
    parser.add_argument('--h5_val_file', default=None, help='the h5 to use it ')
    parser.add_argument('--h5_train_len', default=None, help='')
    parser.add_argument('--h5_val_len', default=None, help='')
    parser.add_argument('--exp_dir', default=None, help='The experiment folder')
    parser.add_argument('--out_dir', default=None, help='The output folder')

    # Data arguments
    parser.add_argument('--img_height', default=None, type=int, help='Image height of the data')
    parser.add_argument('--img_width', default=None, type=int, help='Image width of the data')
    parser.add_argument('--num_channels', default=None, type=int, help='Num of channels of the image of the data')
    parser.add_argument('--num_classes', default=None, type=int, help='Num of classes of the data')
    parser.add_argument('--targets_resize', default=None, type=int, help='In case of experiment_v2 1/targets_resize will be the size of labels in training')

    # Train arguments
    parser.add_argument('--num_epochs', default=2, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--shuffle', action='store_true', help='Flag to shuffle the data while training')
    parser.add_argument('--data_mode', default="experiment", help='Data mode')
    parser.add_argument('--save_every', default=1, type=int, help='save every n epochs')
    parser.add_argument('--test_every', default=1, type=int, help='test every n epochs')
    parser.add_argument('--max_to_keep', default=1, type=int, help='Max checkpoints to keep')
    parser.add_argument('--weighted_loss', action='store_true', help='Flag to use weighted loss or not')
    parser.add_argument('--random_cropping', action='store_true', help='Flag to use random cropping or not')
    parser.add_argument('--freeze_encoder', action='store_true', help='Flag to freeze or train encoding layers')

    # Test arguments

    # Models arguments
    parser.add_argument('--num_groups', default=3, type=int, help='number of groups in group conv.')
    parser.add_argument('--learning_rate', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--learning_decay', default=1e-7, type=float, help='learning decay')
    parser.add_argument('--learning_decay_every', default=100, type=int, help='learning decay_every')
    parser.add_argument('--weight_decay', default=2e-4, type=float, help='weight decay')
    parser.add_argument('--bias', default=0.0, type=float, help='bias')
    parser.add_argument('--pretrained_path', default="", help='The path of pretrained weights')
    parser.add_argument('--batchnorm_enabled', action='store_false', help='Batchnorm Enabled flag')

    # Misc arguments
    parser.add_argument('--verbose', action='store_true', help='verbosity in the code')
    parser.add_argument('--yaml_name', help='Help to know which yaml you have selected')

    # Load the arguments from the configuration file
    yaml_path = os.path.realpath(os.getcwd()) + "/config/experiments_config/" + args.config_path
    if args.config_path:
        with open(yaml_path, 'r') as f:
            parser.set_defaults(**yaml.load(f))

    # parse the parameters
    args = parser.parse_args(unprocessed_args)
    args.config_path = config_path

    # visualize the configuration on the terminal
    visualize_config(args)

    return args
