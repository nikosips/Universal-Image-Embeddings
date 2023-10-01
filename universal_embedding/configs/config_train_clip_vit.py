import ml_collections, os
import jax.numpy as jnp


DATASET_TRAIN_SIZE = { #number of train images per dataset
    'cars': 6346,
    'sop': 48942,
    'inshop': 20897,
    'inat': 273929,
    'met': 397121,
    'gldv2': 1422914,
    'food2k': 472349,
    'rp2k': 188724,
}

ViT_configs = {
    'B/16': {
    "hidden_size" : 768,
    "patches_size" : [16, 16],
    "num_heads" : 12,
    "mlp_dim" : 3072,
    "num_layers" : 12,
    "checkpoint" : 'vit_b16.npy'
  },
}


def get_aggregated_size(datasets):

  size = 0
  
  for dataset in datasets.split(','):
    size += DATASET_TRAIN_SIZE[dataset]

  return size



def get_config():
  """Returns the ViT experiment configuration."""

  config = ml_collections.ConfigDict()

  config.experiment_name = 'universal-embedding-vit'

  # Dataset.
  config.dataset_name = "food2k,cars,sop,inshop,inat,met,gldv2,rp2k"
  config.knn_eval_names = "food2k,cars,sop,inshop,inat,met,gldv2,rp2k"

  config.data_dtype_str = 'float32'
  #config.data_dtype_str = 'bfloat16'
  
  config.disabled_separate_knns = 'train_knn,val_knn,test_knn'
  config.disabled_merged_knns = 'train_knn,test_knn'

  config.dataset_configs = ml_collections.ConfigDict()
  
  config.normalization_statistics = ml_collections.ConfigDict()
  config.normalization_statistics.MEAN_RGB = [0.48145466, 0.4578275, 0.40821073]
  config.normalization_statistics.STDDEV_RGB = [0.26862954, 0.26130258, 0.27577711]
  
  #sampling methods
  #config.sampling_strategy = "dataset_size"
  #config.sampling_strategy = "balanced"
  config.sampling_strategy = "round_robin"
  #config.sampling_strategy = "specialist_top_steps"
  #config.specialist_top_steps = [11070,686,11460,4401,12840,31020,55580,4422] #give the list of weights here

  config.classifier = "joint" 
  #config.classifier = "separate"

  config.count_flops = False #bugged?

  # Model.
  config.model_class = 'clip_vit_with_embedding'  
  config.model_type = "B/16"
  config.model = ml_collections.ConfigDict()
  

  config.model.hidden_size = ViT_configs[config.model_type]["hidden_size"]
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = ViT_configs[config.model_type]["patches_size"]
  config.model.num_heads = ViT_configs[config.model_type]["num_heads"]
  config.model.mlp_dim = ViT_configs[config.model_type]["mlp_dim"]
  config.model.num_layers = ViT_configs[config.model_type]["num_layers"]
  config.model.representation_size = None #we will always use that as None

  #our chosen embedding dimension
  config.model.output_dim = 64 

  config.model.classifier = 'token'
  config.model.attention_dropout_rate = 0.0
  config.model.dropout_rate = 0.0
  
  config.model_dtype_str = 'float32'
  #config.model_dtype_str = 'bfloat16'
  
  #config.model.positional_embedding = 'none'
  config.model.positional_embedding = 'learned_1d'

  #checkpoints
  config.pretrained_ckpt_dir = ''
  config.pretrained_ckpt = os.path.join(config.pretrained_ckpt_dir,ViT_configs[config.model_type]["checkpoint"])
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
  
  config.num_training_epochs = 30

  config.batch_size = 128

  config.steps_per_epoch = (get_aggregated_size(config.dataset_name) // config.batch_size)

  config.eval_batch_size = 1024 
  config.knn_eval_batch_size = 2048

  config.rng_seed = 0 #42 is original
  
  #config.init_head_bias = -10.0 #not used anywhere by us, can we remove it from here?
  
  config.loss = ml_collections.ConfigDict()
  config.loss.m = 0.0
  config.loss.scale = 16

  config.loss.transform_logits_type = 'normface'
  #config.loss.transform_logits_type = 'arcface'
  #config.loss.transform_logits_type = 'cosface'

  config.max_to_keep = 1000

  #log eval steps is how often I do knn? yes, if turned on, and also logging.
  config.log_eval_steps = config.steps_per_epoch

  # Learning rate.
  base_lr = 1e-3

  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant'
  config.lr_configs.base_learning_rate = base_lr

  config.lr_configs.backbone = ml_collections.ConfigDict()

  #that means for 2 epochs we train only the classifier
  config.lr_configs.backbone.frozen_steps = (
      2 * config.steps_per_epoch
  )

  config.lr_configs.backbone.base_learning_rate = base_lr * 1e-2

  # kNN
  config.do_knn = True

  config.do_final_testing = True #final testing datasets are hardcoded for now

  config.save_descriptors = True

  config.extract_only_descrs = False

  config.save_neighbors = False

  # Logging.
  config.write_summary = True
  
  config.checkpoint = True  # Do checkpointing.
  #config.checkpoint = False
  
  #config.checkpoint_steps = config.steps_per_epoch * config.num_training_epochs #save only last model (at the end)
  config.checkpoint_steps = config.steps_per_epoch #save checkpoint after every epoch
  
  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.

  config.do_pcaw = False

  #config.descr_save_path = "."
  config.descr_save_path = None

  config.top_k = 5

  config.eval_dataset_dir = ''
  config.train_dataset_dir = '' 

  config.project_feats_knn = True
  #config.project_feats_knn = False

  config.log_domain_acc = True

  config.log_csv = False

  config.info_files_dir = '' 

  return config