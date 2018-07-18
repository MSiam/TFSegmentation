class OnavosConfigTrain():
    # saving model
    max_to_keep = 2
    load_model = True
    checkpoint_dir = "./checkpoints/"
    save = True
    load = "models/DAVIS16/DAVIS16"  # not needed

    # training
    davis_data_dir = "/home/eren/Data/DAVIS/"
    task = "train"
    dataset = "DAVIS"
    model = "DAVIS16_oneshot"
    batch_size = 1
    batch_size_eval = 1
    log_verbosity = 5
    optimizer = "adam"
    momentum = .9
    freeze_batchnorm = True
    learning_rates = {1: 0.000001, 8: 0.0000005, 12: 0.0000001}
    augmentors_train = ["gamma", "flip"]
    resize_mode_train = ""
    n_epochs = 500
    gpus = [0]

    # eval
    trainsplit = 0
    n_finetune_steps = 0
    ignore_first_and_last_results = True
    resize_mode_val = "unchanged"
    augmentors_val = []
    n_test_samples = 1

    # others
    save_oneshot = True
    save_logits = True
    use_summaries = False
    input_size_val = None
    measures = []
    logfile_path = './log'
