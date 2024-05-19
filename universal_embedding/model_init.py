import logging

import jax.numpy as jnp
from scenic.train_lib import train_utils

from collections import OrderedDict




def initialize_universal_model(
  dataset_dict,
  config,
  model,
  init_rng,
):

  dataset_names = dataset_dict.meta_data['dataset_name'].split(",")
  input_spec = OrderedDict()

  for i,_ in enumerate(dataset_names):

    input_spec_key = (
      ('domain', i),
      ('init', True),
    )

    input_spec[input_spec_key] = [(
      dataset_dict.meta_data['input_shape'],
      dataset_dict.meta_data.get('input_dtype', jnp.float32),
    )]

  (params, model_state, num_trainable_params, gflops) = (
    train_utils.initialize_multitask_model(
      model_def=model.flax_model,
      input_spec=input_spec,          
      config=config,
      rngs=init_rng,
    )
  )

  return (params, model_state, num_trainable_params, gflops)



def load_init_checkpoint(
  config,
  train_state,
  model,
):

  if config.init_ckpt:
    logging.info('Initializing from ckpt %s.', config.init_ckpt)
    ckpt_info = config.init_ckpt.split('/')
    ckpt_dir = '/'.join(ckpt_info[:-1])
    ckpt_num = ckpt_info[-1].split('_')[-1]
    train_state, start_step = train_utils.restore_checkpoint(
        ckpt_dir, train_state, assert_exist=True, step=int(ckpt_num)
    )

  elif config.pretrained_ckpt:

    logging.info('Initializing from ckpt %s.', config.pretrained_ckpt)

    if config.model_class.startswith("clip_vit_with_embedding"):

      train_state = model.load_model_vars(
        train_state, config.pretrained_ckpt,
      )

    elif config.model_class.startswith("vit_with_embedding"):

      train_state = model.load_augreg_params(
        train_state, config.pretrained_ckpt, config.model
      )

  return train_state