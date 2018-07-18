class OnavosConfig():
    # saving model
    max_to_keep = 2
    load_model = True
    checkpoint_dir = "./checkpoints_1stream/"
    save = True
    load = "models/DAVIS16/DAVIS16"  # not needed

    # training
    davis_data_dir = "/home/gemy/work/datasets/DAVIS/"
    task = "online"
    dataset = "DAVIS"
    model = "DAVIS16_oneshot"
    batch_size = 1
    batch_size_eval = 1
    log_verbosity = 5
    optimizer = "adam"
    momentum = .9
    freeze_batchnorm = True
    learning_rates = {1: 0.000001, 8: 0.0000005, 12: 0.0000001}
    augmentors_train = []
    resize_mode_train = ""
    n_epochs = 500
    gpus = [0]

    # eval
    trainsplit = 0
    n_finetune_steps = 0
    ignore_first_and_last_results = True
    resize_mode_val = ""
    augmentors_val = []
    n_test_samples = 1

    # others
    save_oneshot = True
    save_logits = True
    save_results = True
    use_summaries = False
    input_size_val = None
    measures = []
    logfile_path = './log'


    #online adaptation
    n_adaptation_steps= 2
    adaptation_interval= 5
    adaptation_learning_rate= 0.00001
    posterior_positive_threshold= 0.97
    distance_negative_threshold= 220.0
    adaptation_loss_scale= 0.05
    adaptation_erosion_size= 15

    adapt_debug= True
    use_positives=True
    use_negatives=True

