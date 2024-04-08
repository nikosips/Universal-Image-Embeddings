import os,ml_collections

import universal_embedding.info_utils as info_utils


def get_config():
  """Returns the ViT experiment configuration."""

  config = ml_collections.ConfigDict()

  config.experiment_name = 'universal-embedding-vit'

  # Dataset.
  
  config.dataset_name = "food2k,cars,sop,inshop,inat,met,gldv2,rp2k"
  config.knn_eval_names = "food2k,cars,sop,inshop,inat,met,gldv2,rp2k"

  config.data_dtype_str = 'float32'

  config.dataset_configs = ml_collections.ConfigDict()

  #sampling methods

  #config.sampling_strategy = "dataset_size" #strategy that samples according to the length of each dataset
  #config.sampling_strategy = "balanced"
  config.sampling_strategy = "round_robin"
  #config.sampling_strategy = "specialist_top_steps"
  #config.specialist_top_steps = [11070,539,10696,4564,8560,24816,66696,5896] #give the list of weights here

  #config.classifier = "joint"
  config.classifier = "separate"

  config.count_flops = False #bugged?

  # Model.
  #config.model_class = 'vit_with_embedding'
  config.model_class = 'clip_vit_with_embedding'


  if 'clip' in config.model_class:  

    #TODO: put these in the dict of the models
    config.normalization_statistics = ml_collections.ConfigDict()
    config.normalization_statistics.MEAN_RGB = [0.48145466, 0.4578275, 0.40821073]
    config.normalization_statistics.STDDEV_RGB = [0.26862954, 0.26130258, 0.27577711]

    model_configs = info_utils.CLIP_ViT_configs
  
  else:

    model_configs = info_utils.ViT_configs

  #config.model_type = "S/16"
  config.model_type = "B/16"

  #TODO: remove below line
  config.clip = False

  config.model = ml_collections.ConfigDict()

  config.model.hidden_size = model_configs[config.model_type]["hidden_size"]
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = model_configs[config.model_type]["patches_size"]
  config.model.num_heads = model_configs[config.model_type]["num_heads"]
  config.model.mlp_dim = model_configs[config.model_type]["mlp_dim"]
  config.model.num_layers = model_configs[config.model_type]["num_layers"]
  config.model.representation_size = None #we will always use that as None

  config.model.output_dim = 64 #our chosen embedding dimension

  config.model.classifier = 'token'
  config.model.attention_dropout_rate = 0.0
  config.model.dropout_rate = 0.0

  config.model_dtype_str = 'float32'
  #config.model_dtype_str = 'bfloat16'

  #config.model.positional_embedding = 'none'
  config.model.positional_embedding = 'learned_1d'


  #checkpoints
  config.pretrained_ckpt_dir = 'data/models/'
  
  config.pretrained_ckpt = os.path.join(config.pretrained_ckpt_dir, model_configs[config.model_type]["checkpoint"])
  config.init_ckpt = ''

  # Training.
  config.optimizer = 'adam' #its actually adamw that scenic uses if you use weight decay
  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.beta1 = 0.9
  config.optimizer_configs.beta2 = 0.999
  config.optimizer_configs.weight_decay = 1e-6
  config.explicit_weight_decay = None  # No explicit weight decay
  config.l2_decay_factor = None
  # config.max_grad_norm = 1.0
  config.label_smoothing = None

  #config.num_training_epochs = 30
  config.num_training_epochs = 7

  config.batch_size = 128

  config.steps_per_epoch = (info_utils.get_aggregated_size(config.dataset_name) // config.batch_size)

  config.eval_batch_size = 1024
  config.knn_eval_batch_size = 2048

  config.disabled_separate_knns = 'train_knn,test_knn'
  config.disabled_merged_knns = 'train_knn,test_knn'

  config.rng_seed = 0

  #config.init_head_bias = -10.0 #not used anywhere by us, can we remove it from here?

  config.loss = ml_collections.ConfigDict()
  config.loss.m = 0.0
  config.loss.scale = 16

  config.loss.transform_logits_type = 'normface'
  #config.loss.transform_logits_type = 'arcface'
  #config.loss.transform_logits_type = 'cosface'

  config.max_to_keep = 1000

  #number of steps to log knn validation metrics
  config.log_eval_steps = config.steps_per_epoch
  
  #number of steps to log train metrics like loss etc.
  config.log_summary_steps = int(config.steps_per_epoch/10)
  #config.log_summary_steps = int(config.steps_per_epoch)

  # Learning rate.
  base_lr = 1e-3

  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant'
  config.lr_configs.base_learning_rate = base_lr

  config.lr_configs.backbone = ml_collections.ConfigDict()

  
  frozen_epochs = 2 #that means for 2 epochs we train only the classifier
  #frozen_epochs = config.num_training_epochs  # to completely freeze some parts

  config.lr_configs.backbone.frozen_steps = (
      frozen_epochs * config.steps_per_epoch
  )

  config.lr_configs.backbone.base_learning_rate = base_lr * 1e-2

  config.params_early_train = ['output_projection']

  # kNN
  config.do_knn = True

  config.do_final_testing = True

  config.save_descriptors = False

  config.extract_only_descrs = False

  # Logging.
  config.write_summary = True

  config.checkpoint = True
  #config.checkpoint = False  # Do checkpointing.

  config.only_best_checkpoint = True
  #config.only_best_checkpoint = False

  #config.checkpoint_steps = config.steps_per_epoch * config.num_training_epochs #save only last model (at the end)
  config.checkpoint_steps = config.steps_per_epoch #save checkpoint after every epoch

  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.

  config.eval_dataset_dir = ''
  config.train_dataset_dir = ''

  config.project_feats_knn = True

  #config.descr_save_path = "."
  config.descr_save_path = None

  config.save_neighbors = False

  config.top_k = 5

  config.log_domain_acc = True

  config.log_csv = False

  config.info_files_dir = ''

  return config
