import sys, os
import functools

from absl import app
from absl import flags
from absl import logging

from clu import metric_writers
from clu import platform
import flax
import flax.linen as nn
import jax
from ml_collections import config_flags
import tensorflow as tf
from tensorflow.io import gfile

import json

import wandb

from universal_embedding import utils


FLAGS = flags.FLAGS

# These are general flags that are used across most of scenic projects. These
# flags can be accessed via `flags.FLAGS.<flag_name>` and projects can also
# define their own flags in their `main.py`.

config_flags.DEFINE_config_file(
    'config', None, 'Training configuration.', lock_config=False) #needed to change it to false
flags.DEFINE_string('workdir', None, 'Work unit directory.')
flags.DEFINE_string('dataset_service_address', None,
                    'Address of the tf.data service')

##################

flags.DEFINE_string('wandb_project', None, 'Wandb Project Name.')
flags.DEFINE_string('wandb_name', None, 'Wandb Experiment Name.')
flags.DEFINE_string('wandb_group', None, 'Wandb Group Name.')
flags.DEFINE_string('wandb_entity', None, 'Wandb Team Name.')
flags.DEFINE_boolean('use_wandb', False, 'If to log to wandb as well.')

flags.DEFINE_boolean('debug_on_tpu', False, 'Use CPU to debug.')

##################

flags.mark_flags_as_required(['config', 'workdir'])

flax.config.update('flax_use_orbax_checkpointing', False)





def run(main,knn=False,descr_eval=False):

  # Provide access to --jax_backend_target and --jax_xla_backend flags.
  jax.config.config_with_absl()
  app.run(functools.partial(_run_main, main=main,knn=knn,descr_eval=descr_eval))




def _run_main(
  argv,
  *,
  main,
  knn,
  descr_eval,
):
  """Runs the `main` method after some initial setup."""
  
  del argv

  #here we should calculate all dependent parameters.
  #calculate config dependent values based on cmd line config args

  if knn: #case of knn or extract_dir_descriptors

    #dependent value, can't be evaluated in the config
    train_config_params = utils.read_config(os.path.join(FLAGS.config.train_dir,"config.json"))
    train_config_params.update(FLAGS.config)
    FLAGS.config = train_config_params


  if (not knn) and (not descr_eval): #save the training config

    utils.calc_train_dependent_config_values(FLAGS.config) 

    with gfile.GFile(os.path.join(FLAGS.workdir,"config.json"), mode = "w") as f:
      
      os.makedirs(FLAGS.workdir, exist_ok = True)

      json.dump(json.loads(FLAGS.config.to_json_best_effort()),f,indent=4)




  if FLAGS.debug_on_tpu:
    jax.config.update('jax_platform_name', 'cpu')
  
  # Hide any GPUs form TensorFlow. Otherwise, TF might reserve memory and make
  # it unavailable to JAX.
  
  tf.config.experimental.set_visible_devices([], 'GPU')

  # Enable wrapping of all module calls in a named_call for easier profiling:
  nn.enable_named_call()

  if FLAGS.jax_backend_target:
    logging.info('Using JAX backend target %s', FLAGS.jax_backend_target)
    jax_xla_backend = ('None' if FLAGS.jax_xla_backend is None else FLAGS.jax_xla_backend)
    logging.info('Using JAX XLA backend %s', jax_xla_backend)

  logging.info('JAX host: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX devices: %r', jax.devices())

  # Add a note so that we can tell which task is which JAX host.
  # (task 0 is not guaranteed to be the host 0)
  platform.work_unit().set_task_status(
    f'host_id: {jax.process_index()},host_count: {jax.process_count()}',
  )
  
  if jax.process_index() == 0:

    platform.work_unit().create_artifact(
      platform.ArtifactType.DIRECTORY,
      FLAGS.workdir, 
      'Workdir',
    )

  rng = jax.random.PRNGKey(FLAGS.config.rng_seed)

  logging.info('RNG: %s', rng)

  if FLAGS.use_wandb:

    wandb.init(
      project = FLAGS.wandb_project,
      name = FLAGS.wandb_name,
      group = FLAGS.wandb_group,
      entity = FLAGS.wandb_entity,
      sync_tensorboard=True,
      config=FLAGS.config,
    )

  writer = metric_writers.create_default_writer(
    FLAGS.workdir, 
    just_logging=jax.process_index() > 0,
    asynchronous=False,
  )

  try:
    
    main(
      rng=rng, 
      config=FLAGS.config, 
      workdir=FLAGS.workdir, 
      writer=writer,
    )

  except KeyboardInterrupt:
    
    if FLAGS.use_wandb:
      wandb.finish()
    
    sys.exit()

  if FLAGS.use_wandb:
    wandb.finish()