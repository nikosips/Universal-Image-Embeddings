import ml_collections, os


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
    "checkpoint" : 'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'
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

  # Dataset that was used for training.
  config.dataset_name = "food2k,cars,sop,inshop,inat,met,gldv2,rp2k"
  config.knn_eval_names = "food2k,cars,sop,inshop,inat,met,gldv2,rp2k"

  config.data_dtype_str = 'float32'
  #config.data_dtype_str = 'bfloat16'

  config.eval_batch_size = 1024 #batch size for extracting embeddings
  config.knn_eval_batch_size = 1024 #batch size for batch knn search 
  
  #merged eval
  config.disabled_separate_knns = 'train_knn,val_knn,test_knn'
  config.disabled_merged_knns = 'train_knn,val_knn'

  #separate eval
  # config.disabled_separate_knns = 'train_knn,val_knn'
  # config.disabled_merged_knns = 'train_knn,val_knn,test_knn'

  config.dataset_configs = ml_collections.ConfigDict()
  
  config.classifier = "joint" 
  #config.classifier = "separate"

  config.count_flops = False #bugged?

  # Model.
  config.model_class = 'vit_with_embedding'
  
  config.model_type = "B/16"

  config.clip = False

  config.model = ml_collections.ConfigDict()

  #move these to the dictionary of the models
  config.model.hidden_size = ViT_configs[config.model_type]["hidden_size"]
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = ViT_configs[config.model_type]["patches_size"]
  config.model.num_heads = ViT_configs[config.model_type]["num_heads"]
  config.model.mlp_dim = ViT_configs[config.model_type]["mlp_dim"]
  config.model.num_layers = ViT_configs[config.model_type]["num_layers"]
  config.model.representation_size = None #we will always use that as None
  
  config.model.output_dim = 64 #our chosen embedding dimension
  
  config.model.classifier = 'token'
  config.model.attention_dropout_rate = 0.0
  config.model.dropout_rate = 0.0
  
  config.model_dtype_str = 'float32'
  #config.model_dtype_str = 'bfloat16'
  
  #config.model.positional_embedding = 'none'
  config.model.positional_embedding = 'learned_1d'

  config.transform_logits_type = 'normface'

  #checkpoints
  config.pretrained_ckpt_dir = ''
  config.pretrained_ckpt = os.path.join(config.pretrained_ckpt_dir,ViT_configs[config.model_type]["checkpoint"])
  
  # Training.
  config.optimizer = 'adam'
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

  config.rng_seed = 0 

  config.init_head_bias = -10.0 
  config.loss = ml_collections.ConfigDict()
  config.loss.m = 0.0
  config.loss.scale = 16
  config.max_to_keep = 1000

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

  #dir of checkpoints
  config.train_dir = ''
  
  # Logging.

  config.preextracted = False
  #config.preextracted = True

  config.write_summary = True
  
  config.test_pretrained_features = False

  config.extract_only_descrs = False

  config.checkpoint = False 

  config.save_descriptors = True
  #config.save_descriptors = False

  config.debug_eval = False  # Debug mode during eval.
    
  config.eval_dataset_dir = ''
  config.train_dataset_dir = '' 
  
  config.project_feats_knn = True
  #config.project_feats_knn = False

  config.top_k = 5 #top k neighbors to look at
  #config.top_k = 100

  config.knn_start_epoch = 3
  config.knn_end_epoch = 3 #set this to a lower value than start_epoch to not do knn at all

  config.log_csv = False

  config.save_neighbors = False

  config.info_files_dir = ''

  #config.descr_save_path = "."
  config.descr_save_path = None

  return config