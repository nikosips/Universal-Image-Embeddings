import jax.numpy as jnp
import jax
import flax
import optax

from scenic.train_lib import optimizers


def backbone_lr(config):
  
  def lr_fn(step):
    ratio = config['base_learning_rate']
    ratio *= jnp.where(config.frozen_steps >= step, 0.0, 1.0)
    return ratio

  return lr_fn


def get_multioptimizer(
    optimizer_config,
    classifier_lr_fn,
    backbone_lr_fn,
    params,
    config,
):
  """Makes a Flax MultiOptimizer with a separate backbone optimizer."""

  classifier_optim = optimizers.get_optimizer(
      optimizer_config, classifier_lr_fn, params
  )
  backbone_optim = optimizers.get_optimizer(
      optimizer_config, backbone_lr_fn, params
  )

  all_false = jax.tree_util.tree_map(lambda _: False, params)
  
  classifier_traversal = flax.traverse_util.ModelParamTraversal(
      lambda path, param: any(x in path for x in config.params_early_train)
  )

  backbone_traversal = flax.traverse_util.ModelParamTraversal(
      lambda path, param: all(x not in path for x in config.params_early_train)
  )

  classifer_mask = classifier_traversal.update(lambda _: True, all_false)
  backbone_mask = backbone_traversal.update(lambda _: True, all_false)

  return optax.chain(
      optax.masked(backbone_optim, backbone_mask),
      optax.masked(classifier_optim, classifer_mask),
  )