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
  
  'S/16': {
    "hidden_size": 384,
    "patches_size": [16, 16],
    "num_heads": 6,
    "mlp_dim": 1536,
    "num_layers": 12,
    "checkpoint": 'imagenet/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'
  },

  'B/16': {
    "hidden_size": 768,
    "patches_size": [16, 16],
    "num_heads": 12,
    "mlp_dim": 3072,
    "num_layers": 12,
    "checkpoint": 'imagenet/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'
  },

}


CLIP_ViT_configs = {

    'B/16': {
    "hidden_size" : 768,
    "patches_size" : [16, 16],
    "num_heads" : 12,
    "mlp_dim" : 3072,
    "num_layers" : 12,
    "checkpoint" : 'clip/vit_b16.npy'
  },

}


def get_aggregated_size(datasets):

  size = 0

  for dataset in datasets.split(','):
    size += DATASET_TRAIN_SIZE[dataset]

  return size