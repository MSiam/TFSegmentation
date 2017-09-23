class OnavosConfig():
    # saving model
    max_to_keep = 2
    load_model = True
    checkpoint_dir = "../checkpoints/chk"

    davis_data_dir = "/home/gemy/work/datasets/DAVIS/"
    model = "DAVIS16_oneshot"
    task = "eval"
    dataset = "DAVIS"
    batch_size = 1
    batch_size_eval = 1
    log_verbosity = 5
    gpus = [0]
    optimizer = "adam"
    freeze_batchnorm = True
    save = True
    num_epochs = 20

    trainsplit = 0

    load = "models/DAVIS16/DAVIS16"
    n_finetune_steps = 50
    learning_rates = "{1: 0.000003}"
    save_oneshot = True
    save_logits = True
    batch_size=1
    augmentors_train = []

    resize_mode_train = ""
    use_summaries=False
    input_size_val=None
    resize_mode_val = ""
    augmentors_val = []
    n_test_samples = 1
