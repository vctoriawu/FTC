def get_config():
    """Get the hyperparameter configuration."""
    config = {}
    
    config['mode'] = "train"
    config['use_wandb'] = False
    config['use_cuda'] = True
    config['num_workers'] = 8
    config['log_dir'] = "/AS_clean/tuft_fs/logs"
    config['log_dir'] = "/workspace/miccai2024_savedmodels/FTC/logs"
    config['model_load_dir'] = "FTC_image_tmed" 
    config['best_model_dir'] = "/workspace/miccai2024_savedmodels/FTC/logs/FTC_image_tmed" 

    # Hyperparameters for dataset. 
    config['view'] = 'plaxpsax' # all/plax/psax
    config['flip_rate'] = 0.3
    config['label_scheme_name'] = 'mod_severe'
    config['view_scheme_name'] = 'three_class'
    # must be compatible with number of unique values in label scheme
    # will be automatic in future update
    config['num_classes_diagnosis'] = 4
    config['num_classes_view'] = 2

    # Hyperparameters for models.
    config['model'] = "FTC_image_tmed" # wideresnet
    config['pretrained'] = False
    config['restore'] = False
    config['loss_type'] = 'cross_entropy' # cross_entropy/evidential/laplace_cdf/SupCon/SimCLR

    # Hyperparameters for training.
    config['batch_size'] = 4
    config['num_epochs'] = 500
    config['lr'] = 0.00007  #1e-4 for Resnet2+1D, 1e-5 for FTC
    config['sampler'] = 'random' # imbalanaced sampling based on AS/bicuspid/random
 
    return config