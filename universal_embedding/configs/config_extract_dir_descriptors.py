import ml_collections, os


#TODO: change image size here

def get_config():

  """Returns the ViT experiment configuration."""

  config = ml_collections.ConfigDict()

  config.train_dir = "" #dir containing checkpoint and train config
  config.base_dir = ""

  # kNN configs

  #values below here overwrite those of training config that are the default ones

  #config.embedd_to_eval = "projected"
  config.embedd_to_eval = "backbone_out"

  config.knn_eval_names = "food2k,cars,sop,inshop,inat,met,gldv2,rp2k"
  
  config.disabled_separate_knns = 'train_knn,val_knn'
  config.disabled_merged_knns = 'train_knn,val_knn'
  
  config.eval_batch_size = 1024
  config.knn_eval_batch_size = 2048

  config.preextracted = False
  #config.preextracted = True

  config.write_summary = True
  #config.write_summary = False
  
  config.test_pretrained_features = False

  config.extract_only_descrs = False

  #config.save_descriptors = True
  config.save_descriptors = False

  config.debug_eval = False  # Debug mode during eval.
    
  config.eval_dataset_dir = ''
  config.train_dataset_dir = '' 
  
  config.project_feats_knn = True
  #config.project_feats_knn = False

  config.top_k = 5 #top k neighbors to look at
  #config.top_k = 100

  #if you were saving only best checkpoint
  config.only_best_knn = True
  #config.only_best_knn = False

  #if only_best_knn is True, below epochs are not taken into account
  config.knn_start_epoch = 3
  config.knn_end_epoch = 7 #set this to a lower value than start_epoch to not do knn at all

  config.log_csv = False

  config.save_neighbors = False

  config.info_files_dir = ''

  config.descr_save_path = "." #which one is the workdir?
  #config.descr_save_path = None

  return config