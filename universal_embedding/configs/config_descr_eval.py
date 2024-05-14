import ml_collections, os
from universal_embedding import utils


#TODO: make sure this works properly with new app.py


def get_config():

  """Returns the ViT experiment configuration."""

  config = ml_collections.ConfigDict()

  config.descr_path= ''

  
  # kNN configs
  config.descr_eval=True
  
  config.rng_seed = 0

  config.knn_eval_names = "food2k,cars,sop,inshop,inat,met,gldv2,rp2k"

  config.disabled_separate_knns = 'train_knn,val_knn'
  config.disabled_merged_knns = 'train_knn,val_knn'

  config.eval_batch_size = 1024
  config.knn_eval_batch_size = 2048

  config.preextracted = False
  #config.preextracted = True

  #config.write_summary = True
  config.write_summary = False
  
  config.test_pretrained_features = False

  config.extract_only_descrs = False

  config.save_descriptors = True
  #config.save_descriptors = False

  config.debug_eval = False  # Debug mode during eval.
    
  config.eval_dataset_dir = ''
  config.train_dataset_dir = '' 
  
  config.project_feats_knn = True
  #config.project_feats_knn = False

  config.top_k = 5 #top k neighbors to look at
  #config.top_k = 100

  config.only_best_knn = True
  #config.only_best_knn = False

  config.knn_start_epoch = 3
  config.knn_end_epoch = 7 #set this to a lower value than start_epoch to not do knn at all

  config.log_csv = False

  config.save_neighbors = False

  config.info_files_dir = ''

  #config.descr_save_path = "."
  config.descr_save_path = None

  return config