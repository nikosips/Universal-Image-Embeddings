"""Script for Knn evalulation."""
import functools

from clu import metric_writers

from absl import logging
from flax import jax_utils
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app
from scenic.train_lib import train_utils

import os
import sys

if sys.version_info.major == 3 and sys.version_info.minor >= 10:

  from collections.abc import MutableMapping
else:
  from collections import MutableMapping

from universal_embedding import classification_with_knn_eval_trainer
from universal_embedding import datasets
from universal_embedding import knn_utils
from universal_embedding import models




def knn_evaluate(
  rng: jnp.ndarray,
  config: ml_collections.ConfigDict,
  workdir: str,
  writer: metric_writers.MetricWriter,
) -> None:

  lead_host = jax.process_index() == 0

  #dataset used for training
  dataset_dict = datasets.get_training_dataset_new(config)

  model_cls = models.MODELS[config.model_class]
  model = model_cls(config, dataset_dict.meta_data)

  rng, init_rng = jax.random.split(rng)
  
  (params, model_state, _, _) = (
      train_utils.initialize_model(
        model_def=model.flax_model,
        input_spec=[(
            dataset_dict.meta_data['input_shape'],
            dataset_dict.meta_data.get('input_dtype', jnp.float32),
        )],
        config=config,
        rngs=init_rng,
      )
  )
  
  train_state = train_utils.TrainState(
    params=params, 
    model_state=model_state
  )


  if config.pretrained_ckpt:
    
    logging.info('Initializing from ckpt %s.', config.pretrained_ckpt)

    if config.model_class == "clip_vit_with_embedding":

      train_state = model.load_model_vars(
          train_state, config.pretrained_ckpt,
      )

    elif config.model_class == "vit_with_embedding":

      train_state = model.load_augreg_params(
          train_state, config.pretrained_ckpt, config.model
      )
    
  train_state = jax_utils.replicate(train_state)

  del params

  #project feats or not
  representation_fn_knn = functools.partial(
    classification_with_knn_eval_trainer.representation_fn_eval,
    flax_model = model.flax_model, 
    project_feats = config.project_feats_knn
  )

  knn_eval_batch_size = config.get('knn_eval_batch_size') or config.batch_size

  knn_evaluator = knn_utils.KNNEvaluator(
    config,
    representation_fn_knn,
    knn_eval_batch_size,
    config.get("extract_only_descrs",False),
  )


  train_dir = config.get('train_dir')

  if config.test_pretrained_features:

    knn_utils.knn_step(
      knn_evaluator,
      train_state,
      config,
      train_dir,
      0,
      writer,
      config.preextracted,
    )

  for epoch in range(config.knn_start_epoch,config.knn_end_epoch+1):

    step = epoch * config.steps_per_epoch

    print(f"step: {step}")

    if not config.preextracted:
      ckpt_file = os.path.join(train_dir,str(step))  
      ckpt_info = ckpt_file.split('/')
      ckpt_dir = '/'.join(ckpt_info[:-1])
      ckpt_num = ckpt_info[-1].split('_')[-1]

      try:

        train_state, _ = train_utils.restore_checkpoint(
          ckpt_dir, 
          train_state, 
          assert_exist=True, 
          step=int(ckpt_num),
        )
        
      except:

        sys.exit("no checkpoint found")

      train_state = jax_utils.replicate(train_state)

    else:

      train_state = None

    knn_utils.knn_step(
      knn_evaluator,
      train_state,
      config,
      train_dir,
      step,
      writer,
      config.preextracted,
    )

  train_utils.barrier_across_hosts()



if __name__ == '__main__':
  app.run(main=knn_evaluate)