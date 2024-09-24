def get_config():
    """Get the hyperparameter configuration."""
    config = {}
    
    config['experiment_name'] = "4_class_coteaching_logreg_dfr" 
    config['seed'] = 42
    config['mode'] = "train"
    config["do_train"] = False
    config["do_comprehensive_test"] = True
    config['use_wandb'] = False
    config['use_cuda'] = True
    config['save_videos'] = False 
    config['dfr_result_path'] = "/workspace/miccai2024_savedmodels/FTC/logs/dfr_logs/"
    config['best_model_dir'] = "/workspace/miccai2024_savedmodels/FTC/logs/dfr_logs/4_class_coteaching"
    config['pretrained_model_dir'] = "/workspace/miccai2024_savedmodels/FTC/logs/4_class_coteaching/best_model_acc.pth"
    
    #DFR configs
    config["n_places"] = 2 #The unique values for the confounding attribute (bicuspid, not bicuspid)
    config["classifier"] = "logistic_reg" #logistic_reg/mlp
    config["balance_dfr_val"] = True #Subset validation to have equal groups for DFR(Val)
    config["notrain_dfr_val"] = True #Do not add train data for DFR(Val)
    config["tune_class_weights_dfr_train"] = False #do not use if you do not want to tune the class weights

    # Hyperparameters for dataset. 
    config['view'] = 'all' # all/plax/psax
    config['flip_rate'] = 0.3
    config['label_scheme_name'] = 'all'
    # must be compatible with number of unique values in label scheme
    # will be automatic in future update
    # number of AS classes
    config['num_classes'] = 4


    #Hyperaparameters for tabular dataset.
    config['use_tab'] = True
    config['scale_feats'] = True
    config['num_ex'] = None
    config['drop_cols'] = []
    config['categorical_cols'] = []
    

    
    # Hyperparameters for bicuspid valve branch
    # config['bicuspid_weight'] = 1.0 # default is 1.0
    
    # Hyperparameters for Contrastive Learning
    config['cotrastive_method'] = 'CE' #'CE'/'SupCon'/'SimCLR'/Linear'
    config['feature_dim'] = 1024
    config['temp'] = 0.1

    # Hyperparameters for models.
    config['model'] = "FTC_TAD" # r2plus1d_18/x3d/resnet50/slowfast/tvn/FTC
    config['pretrained'] = False
    config['restore'] = True
    config['loss_type'] = 'cross_entropy' # cross_entropy/evidential/laplace_cdf/SupCon/SimCLR
    config['abstention'] = False
    config["coteaching"] = False
    config['multimodal'] = "fttrans" # clip/mlp/fttrans
    config['l2_reg_method'] = 'no_reg' # reg_on_vid_only/reg_on_all/no_reg
    config['lr_scheduler'] = 'cosine_annealing' # reduce_on_plateau/cosine_annealing
    config["loss_vid_weight"] = 1
    config["loss_tab_weight"] = 1
    
    config["frame_attention_loss"] = 'none' # kl_div/cosine_sim/none
    config["frame_att_loss_weight"] = 0.5 #if frame_attention_loss='none', this will have no impact
    config["video_entropy"] = True
    config["multimodal_att_entropy"] = False

    # Hyperparameters for training.
    config['batch_size'] = 16
    config['num_epochs'] = 20 #110
    config['lr'] = 1e-5  #1e-4 for Resnet2+1D, 1e-5 for FTC
    config['sampler'] = 'AS' # imbalanced sampling based on AS/bicuspid/random
 
    return config
