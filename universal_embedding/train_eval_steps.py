import functools
from typing import Any, Callable, Dict, Tuple, Optional

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from clu import platform

import numpy as np

import flax
from flax import jax_utils


import flax.linen as nn
import jax
from jax.example_libraries.optimizers import clip_grads
import jax.numpy as jnp
import jax.profiler
import ml_collections
import optax

from scenic.train_lib import train_utils


# Aliases for custom types:
Batch = Dict[str, jnp.ndarray]
MetricFn = Callable[
    [jnp.ndarray, Dict[str, jnp.ndarray]], Dict[str, Tuple[float, int]]
]
LossFn = Callable[[jnp.ndarray, Batch, Optional[jnp.ndarray]], float]
LrFn = Callable[[jnp.ndarray], jnp.ndarray]



def train_step(
    train_state: train_utils.TrainState,
    batch: Batch,
    batch_domain_idx: int, #always clean batches
    *,
    flax_model: nn.Module,
    loss_fn: LossFn,
    classifier_lr_fn: LrFn,
    backbone_lr_fn: LrFn,
    metrics_fn: MetricFn,
    config: ml_collections.ConfigDict,
    debug: Optional[bool] = False,
) -> Tuple[
    train_utils.TrainState, Dict[str, Tuple[float, int]], Dict[str, Any]
]:
  """Runs a single step of training.

  Given the state of the training and a batch of data, computes
  the loss and updates the parameters of the model.

  Note that in this code, the buffers of the first (train_state) and second
  (batch) arguments are donated to the computation.

  Args:
    train_state: The state of training including the current global_step,
      model_state, rng, params, and optimizer. The buffer of this argument can
      be donated to the computation.
    batch: A single batch of data. The buffer of this argument can be donated to
      the computation.
    flax_model: A Flax model.
    loss_fn: A loss function that given logits, a batch, and parameters of the
      model calculates the loss.
    classifier_lr_fn: The learning rate fn used for the logging the classifier
      learning rate.
    backbone_lr_fn: The learning rate fn used for the logging the backbone
      learning rate.
    metrics_fn: A metrics function that given logits and batch of data,
      calculates the metrics as well as the loss.
    config: Configurations of the experiment.
    debug: Whether the debug mode is enabled during training. `debug=True`
      enables model specific logging/storing some values using
      jax.host_callback.

  Returns:
    Updated state of training and computed metrics and some training logs.
  """

  training_logs = {}
  new_rng, rng = jax.random.split(train_state.rng)

  # Bind the rng to the host/device we are on.
  dropout_rng = train_utils.bind_rng_to_host_device(
      rng, axis_name='batch', bind_to='device'
  )

  def training_loss_fn(params):
  
    variables = {'params': params, **train_state.model_state}
    
    outputs, new_model_state = flax_model.apply(
      variables,
      batch['inputs'],
      domain = batch_domain_idx,
      mutable=['batch_stats'],
      train=True,
      rngs={'dropout': dropout_rng},
      debug=debug,
    )

    loss = loss_fn(
      outputs, 
      batch, 
      variables['params'], 
    )
    
    return loss, (new_model_state, outputs)

  compute_gradient_fn = jax.value_and_grad(training_loss_fn, has_aux=True)
  (train_cost, (new_model_state, outputs)), grad = compute_gradient_fn(
      train_state.params
  )

  del train_cost

  # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
  grad = jax.lax.pmean(grad, axis_name='batch')

  if config.get('max_grad_norm') is not None:
    grad = clip_grads(grad, config.max_grad_norm)

  updates, new_opt_state = train_state.tx.update(
      grad, train_state.opt_state, train_state.params
  )
  new_params = optax.apply_updates(train_state.params, updates)

  training_logs['l2_grads'] = jnp.sqrt(
      sum([jnp.vdot(g, g) for g in jax.tree_util.tree_leaves(grad)])
  )
  
  ps = jax.tree_util.tree_leaves(new_params)
  training_logs['l2_params'] = jnp.sqrt(sum([jnp.vdot(p, p) for p in ps]))
  us = jax.tree_util.tree_leaves(updates)
  training_logs['l2_updates'] = jnp.sqrt(sum([jnp.vdot(u, u) for u in us]))
  # TODO(dehghani): Can we get this from the optimizer instead?
  training_logs['classifier_lr'] = classifier_lr_fn(train_state.global_step)
  training_logs['backbone_lr'] = backbone_lr_fn(train_state.global_step)

  metrics = metrics_fn(
    outputs, 
    batch, 
  )
  
  new_train_state = train_state.replace(
    global_step=train_state.global_step + 1,
    opt_state=new_opt_state,
    params=new_params,
    model_state=new_model_state,
    rng=new_rng,
  )

  return new_train_state, metrics, training_logs



def representation_fn_eval(
    train_state: train_utils.TrainState,
    batch: Batch,
    *,
    flax_model: nn.Module,
    project_feats = True,
    gather_to_host: bool = True,
    config: ml_collections.ConfigDict = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Feeds the inputs to the model and returns their representations.

  Args:
    train_state: TrainState, the state of training including the current
      global_step, model_state, rng, and optimizer. The buffer of this argument
      can be donated to the computation.
    batch: A single batch of data from the dataset.
    flax_model: A Flax model.
    gather_to_host: Whether to gather results from all devices to the host,
      rather than leaving them distributed.

  Returns:
    Representation learned by the model for the given inputs and the labels and
    masks. If `gather_to_host` is True, these are collected from all hosts.
  """
  variables = {'params': train_state.params, **train_state.model_state}

  outputs = flax_model.apply(
    variables, 
    batch['inputs'],
    domain = -1, #domain agnostic feature extraction
    train=False,
    return_feats=True, #do not use classifier
    debug=False,
    project_feats = project_feats,
  )

  embedding = outputs["embeddings"][config.embedd_to_eval]

  if gather_to_host:
    embedding = jax.lax.all_gather(embedding, 'batch')
    batch = jax.lax.all_gather(batch, 'batch')
  

  return embedding, batch['batch_mask']