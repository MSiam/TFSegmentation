import os


def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def create_exp_dirs(args):
    """
    Create experiment and out dirs
    :param args: Arguments of the program
    :return: args , The new one which contains all needed dirs
    """
    args.data_dir = os.path.realpath(os.getcwd()) + "/data/" + args.data_dir + "/"
    args.exp_dir = os.path.realpath(os.getcwd()) + "/experiments/" + args.exp_dir + "/"
    args.out_dir = os.path.realpath(os.getcwd()) + "/out/" + args.out_dir + "/"
    args.summary_dir = args.exp_dir + 'summaries/'
    args.checkpoint_dir = args.exp_dir + 'checkpoints/'
    args.checkpoint_best_dir = args.exp_dir + 'checkpoints/best/'
    args.npy_dir = args.out_dir + 'npy/'
    args.metrics_dir = args.out_dir + 'metrics/'
    args.imgs_dir = args.out_dir + 'imgs/'

    dirs_to_be_created = [args.checkpoint_dir,
                          args.checkpoint_best_dir,
                          args.summary_dir,
                          args.npy_dir,
                          args.metrics_dir,
                          args.imgs_dir]
    # Create the dirs if it is not exist
    create_dirs(dirs_to_be_created)

    return args
