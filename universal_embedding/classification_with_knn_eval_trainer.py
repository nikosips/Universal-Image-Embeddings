"""Training script and knn evalulation."""

import functools
from typing import Any, Callable, Dict, Tuple, Optional, Type

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

from scenic.train_lib import lr_schedules
from scenic.train_lib import optimizers
from scenic.train_lib import train_utils

from universal_embedding import knn_utils
from universal_embedding import utils
from universal_embedding import sampling_utils




# Aliases for custom types:
Batch = Dict[str, jnp.ndarray]
MetricFn = Callable[
    [jnp.ndarray, Dict[str, jnp.ndarray]], Dict[str, Tuple[float, int]]
]
LossFn = Callable[[jnp.ndarray, Batch, Optional[jnp.ndarray]], float]
LrFn = Callable[[jnp.ndarray], jnp.ndarray]



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


def train_step(
    train_state: train_utils.TrainState,
    batch: Batch,
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
    
    logits, new_model_state = flax_model.apply(
      variables,
      batch['inputs'],
      mutable=['batch_stats'],
      train=True,
      rngs={'dropout': dropout_rng},
      debug=debug,
    )

    loss = loss_fn(
      logits, 
      batch, 
      variables['params'], 
      classifier = config.classifier,
    )
    
    return loss, (new_model_state, logits)

  compute_gradient_fn = jax.value_and_grad(training_loss_fn, has_aux=True)
  (train_cost, (new_model_state, logits)), grad = compute_gradient_fn(
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
    logits, 
    batch, 
    classifier = config.classifier,
    )
  
  new_train_state = train_state.replace(
      global_step=train_state.global_step + 1,
      opt_state=new_opt_state,
      params=new_params,
      model_state=new_model_state,
      rng=new_rng,
  )

  return new_train_state, metrics, training_logs, logits



def representation_fn_eval(
    train_state: train_utils.TrainState,
    batch: Batch,
    *,
    flax_model: nn.Module,
    project_feats = True,
    gather_to_host: bool = True,
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

  embedding = flax_model.apply(
    variables, 
    batch['inputs'],
    train=False,
    return_feats = True,
    debug=False, 
    project_feats = project_feats,
  )

  if gather_to_host:
    embedding = jax.lax.all_gather(embedding, 'batch')
    batch = jax.lax.all_gather(batch, 'batch')
  

  return embedding, batch['batch_mask']



def train(
    *,
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    model_cls: Any,
    dataset_dict: Dict,   
    workdir: str,
    writer: metric_writers.MetricWriter,
) -> Tuple[train_utils.TrainState, Dict[str, Any], Dict[str, Any]]:
  
  """Main training loop lives in this function.

  Given the model class and dataset, it prepares the items needed to run the
  training, including the TrainState.

  Args:
    rng: Jax rng key.
    config: Configurations of the experiment.
    model_cls: Model class; A model has a flax_module, a loss_fn, and a
      metrics_fn associated with it.
    dataset: The dataset that has train_iter, eval_iter, meta_data, and
      optionally, test_iter.
    workdir: Directory for checkpointing.
    writer: CLU metrics writer instance.

  Returns:
    train_state that has the state of training (including current
      global_step, model_state, rng, and the optimizer), train_summary
      and eval_summary which are dict of metrics. These outputs are used for
      regression testing.
  """

  lead_host = jax.process_index() == 0

  # Build the loss_fn, metrics, and flax_model.
  model = model_cls(config, dataset_dict.meta_data)

  # Initialize model. (initialize it's parameters)
  rng, init_rng = jax.random.split(rng)

  #keep in mind that model is initialized in one device 
  (params, model_state, num_trainable_params, gflops) = (
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

  # Create optimizer.
  classifier_lr_fn = lr_schedules.get_learning_rate_fn(config)
  backbone_lr_fn = backbone_lr(config.lr_configs.backbone)
  optimizer_config = optimizers.get_optax_optimizer_config(config)
  
  # If the config is already an optax-compatible config, better call directly:
  #   optimizers.get_optimizer(config.optimizer_configs, lr_fn)

  tx = get_multioptimizer(
      optimizer_config,
      classifier_lr_fn,
      backbone_lr_fn,
      params=params,
      config=config,
  )
  
  # We jit this, such that the arrays that are created on the same device as the
  # input is, in this case the CPU. Else they'd be on device[0].
  opt_state = jax.jit(tx.init, backend='cpu')(params)

  rng, train_rng = jax.random.split(rng)

  # Create chrono class to track and store training statistics and metadata:
  chrono = train_utils.Chrono(warmup = 1)

  train_state = train_utils.TrainState(
      global_step=0,
      opt_state=opt_state,
      tx=tx,
      params=params,
      model_state=model_state,
      rng=train_rng,
      metadata={'chrono': chrono.save()},
  )
  
  start_step = train_state.global_step

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

    if config.model_class == "clip_vit_with_embedding":

      train_state = model.load_model_vars(
          train_state, config.pretrained_ckpt,
      )

    elif config.model_class == "vit_with_embedding":

      train_state = model.load_augreg_params(
          train_state, config.pretrained_ckpt, config.model
      )

  chrono.load(train_state.metadata['chrono'])
  train_state = train_state.replace(metadata={})

  # Replicate the optimizer, state, and rng.
  train_state = jax_utils.replicate(train_state)

  del params  # Do not keep a copy of the initial params.

  # Calculate the total number of training steps.
  total_steps, steps_per_epoch = train_utils.get_num_training_steps(
      config, dataset_dict.meta_data
  )

  #initialize the sampler that creates the data sampling scheme
  sampler = sampling_utils.Sampler(config,dataset_dict,total_steps)

  assert len(sampler.ds_indices_per_step) == total_steps


  train_step_pmapped = jax.pmap(
      functools.partial(
          train_step,
          flax_model=model.flax_model,
          loss_fn=model.loss_function,
          classifier_lr_fn=classifier_lr_fn,
          backbone_lr_fn=backbone_lr_fn,
          metrics_fn=model.get_metrics_fn('train'),
          config=config,
          debug=config.debug_train,
      ),
      axis_name='batch',
      # We can donate both buffers of train_state and train_batch.
      donate_argnums=(0, 1),
  )

  representation_fn_knn = functools.partial(
      representation_fn_eval, 
      flax_model = model.flax_model, 
      project_feats = config.project_feats_knn, #project embedding or not
  )

  knn_eval_batch_size = config.get('knn_eval_batch_size') or config.batch_size
  
  knn_evaluator = knn_utils.KNNEvaluator(
      config,
      representation_fn_knn, 
      knn_eval_batch_size,
      config.get("extract_only_descrs",False),
  )

  log_eval_steps = config.get('log_eval_steps') or steps_per_epoch
  
  if not log_eval_steps:
    raise ValueError("'log_eval_steps' should be specified in the config.")
  
  checkpoint_steps = config.get('checkpoint_steps') or log_eval_steps
  log_summary_steps = config.get('log_summary_steps') or log_eval_steps

  train_metrics, extra_training_logs = [], []
  train_summary, eval_summary = None, None

  train_domain_metrics = {}

  chrono.inform(start_step, total_steps, config.batch_size, steps_per_epoch)
  logging.info('Starting training loop at step %d.', start_step + 1)
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=total_steps, writer=writer
  )

  def write_note(note):
    if lead_host:
      platform.work_unit().set_notes(note)

  hooks = []
  if lead_host:
    hooks.append(report_progress)
  if config.get('xprof', True) and lead_host:
    hooks.append(periodic_actions.Profile(num_profile_steps=5, logdir=workdir))

  if start_step == 0:
    step0_log = {'num_trainable_params': num_trainable_params}
    if gflops:
      step0_log['gflops'] = gflops
    writer.write_scalars(1, step0_log)

  write_note(f'First step compilations...\n{chrono.note}')
  write_note(f'Using classifier: {config.classifier}')

  best_val_step = 0
  best_average_common_val_knn_top_1 = 0

  for step in range(start_step + 1, total_steps + 1):

    with jax.profiler.StepTraceAnnotation('train', step_num=step):

      train_batch,batch_dataset_idx,batch_dataset_name = sampler.get_next_train_batch(step)
      
      train_state, t_metrics, t_logs, logits = train_step_pmapped(
          train_state, train_batch,
      )

      if config.log_domain_acc:

        t_metrics_unreplicated = jax.tree_util.tree_map(jax_utils.unreplicate, t_metrics)

        if batch_dataset_name not in train_domain_metrics:
          train_domain_metrics[batch_dataset_name] = {"loss": [], "accuracy": []}

        #averaging across batch
        batch_loss = t_metrics_unreplicated["loss"][0]/t_metrics_unreplicated["loss"][1]
        batch_accuracy = t_metrics_unreplicated["prec@1"][0]/t_metrics_unreplicated["prec@1"][1]
        
        train_domain_metrics[batch_dataset_name]["loss"].append(batch_loss)
        train_domain_metrics[batch_dataset_name]["accuracy"].append(batch_accuracy)

      # This will accumulate metrics in TPU memory up to the point that we log
      # them. This is no problem for small metrics but may be a problem for
      # large (e.g. segmentation) metrics. An alternative is to set
      # `log_summary_steps` to a small number, or to use
      # `train_utils.unreplicate_and_get` here instead of right before writing
      # summaries, but that means in each step, we have data transfer between
      # tpu and host, which might slow down the training.

      train_metrics.append(t_metrics)
      # Additional training logs: learning rate:
      t_logs = jax.tree_util.tree_map(jax_utils.unreplicate, t_logs)
      extra_training_logs.append(t_logs)
    
    for h in hooks:
      h(step)

    # Below are once-in-a-while ops -> pause.
    ###################### LOG TRAIN SUMMARY ########################
    if (
        (step % log_summary_steps == 1)
        or (step == total_steps)
        or (lead_host and chrono.warmup)
    ):
      
      chrono.pause()
      
      if lead_host:
        chrono.tick(step, writer, write_note)
      
      # train_metrics is list of a dictionaries of metrics, where the shape of
      # the metrics[key] is [n_local_devices]. However, because metric functions
      # have a psum, we have already summed across the whole sharded batch, and
      # what's returned is n_local_devices copies of the same summed metric.
      # So we do unreplicate and fetch them to host using `unreplicate_and_get`.

      train_summary = train_utils.log_train_summary(
          step=step,
          train_metrics=jax.tree_util.tree_map(
              train_utils.unreplicate_and_get, train_metrics
          ),
          extra_training_logs=jax.tree_util.tree_map(
              jax.device_get, extra_training_logs
          ),
          writer=writer,
      )

      if config.log_domain_acc:

        epoch_domain_accuracy = {}
        epoch_domain_loss = {}

        for dataset_name in train_domain_metrics:
          
          dataset_loss = train_domain_metrics[dataset_name]["loss"]
          dataset_accuracy = train_domain_metrics[dataset_name]["accuracy"]
          
          epoch_domain_accuracy[f'train_domain_accuracy/{dataset_name}'] = round(np.mean(dataset_accuracy),3)
          epoch_domain_loss[f'train_domain_loss/{dataset_name}'] = round(np.mean(dataset_loss),3)

        writer.write_scalars(step, epoch_domain_accuracy)
        writer.write_scalars(step, epoch_domain_loss)


      # Reset metric accumulation for next evaluation cycle.
      train_metrics, extra_training_logs = [], []
      train_domain_metrics = {}

      chrono.resume()

    ##################### CHECKPOINTING ###################
    if (
        (step % checkpoint_steps == 0 and step > 0) or (step == total_steps)
    ) and config.checkpoint and not config.only_best_checkpoint:
      chrono.pause(wait_for=(train_state.params, train_state.opt_state))

      with report_progress.timed('checkpoint'):
        
        # Sync model st date across replicas.
        train_state = train_utils.sync_model_state_across_replicas(train_state)
        
        if lead_host:
          # Take the first replica.
          unrep_train_state = jax_utils.unreplicate(train_state)
          metadata = unrep_train_state.metadata
          metadata['chrono'] = chrono.save()
          unrep_train_state.replace(metadata=metadata)
          train_utils.save_checkpoint(
              workdir,
              unrep_train_state,
              max_to_keep=config.get('max_to_keep', 10),
              overwrite=True
          )
          del unrep_train_state
      
      chrono.resume()  # Un-pause now.

    ################### KNN EVALUATION #######################

    if (step % log_eval_steps == 0) or (step == total_steps):
      chrono.pause(wait_for=(train_state.params))
      
      do_knn = config.get('do_knn')

      if do_knn:

        with report_progress.timed('knn'):
          
          results = knn_utils.knn_step(
            knn_evaluator,
            train_state,
            config,
            workdir,
            step,
            writer,
            load_descrs = False,
          )

          #remember best val epoch

          #temporary fix using exceptions
          try:
            epoch_average_common_val_knn_top_1 = results[0]['average:common:val_knn:top_1']
          except:
            epoch_average_common_val_knn_top_1 = results[0]['average:separate:val_knn:top_1']

          if epoch_average_common_val_knn_top_1 > best_average_common_val_knn_top_1:
            best_val_step = step
            best_average_common_val_knn_top_1 = epoch_average_common_val_knn_top_1
          
            if config.checkpoint and config.only_best_checkpoint:
              
              #chrono.pause(wait_for=(train_state.params, train_state.opt_state))

              with report_progress.timed('checkpoint'):
                
                # Sync model st date across replicas.
                train_state = train_utils.sync_model_state_across_replicas(train_state)
                
                if lead_host:
                  # Take the first replica.
                  unrep_train_state = jax_utils.unreplicate(train_state)
                  metadata = unrep_train_state.metadata
                  metadata['chrono'] = chrono.save()
                  unrep_train_state.replace(metadata=metadata)  # pytype: disable=attribute-error
                  #import ipdb; ipdb.set_trace()
                  utils.save_best_checkpoint(
                      workdir,
                      unrep_train_state,
                  )
                  del unrep_train_state


          writer.write_scalars(step, {'best_val_step':best_val_step})
          writer.write_scalars(step, {'best_average_common_val_knn_top_1':best_average_common_val_knn_top_1})

      writer.flush()
      chrono.resume()


  #Do common testing on best val step checkpoint by loading the corresponding checkpoint
  if config.do_final_testing:

    if config.only_best_checkpoint:
      train_state, _ = train_utils.restore_checkpoint(
          workdir, train_state, assert_exist=True, step=-1,
      )

    else:
      train_state, _ = train_utils.restore_checkpoint(
          workdir, train_state, assert_exist=True, step=int(best_val_step),
      )

    # Replicate the optimizer, state, and rng.
    train_state = jax_utils.replicate(train_state)

    config.disabled_merged_knns = "train_knn,val_knn"
    config.knn_eval_names = "food2k,cars,sop,inshop,inat,met,gldv2,rp2k"


    if config.only_best_checkpoint:
      results = knn_utils.knn_step(
        knn_evaluator,
        train_state,
        config,
        workdir,
        -1,
        writer,
        load_descrs=False,
      )
    else:
      results = knn_utils.knn_step(
        knn_evaluator,
        train_state,
        config,
        workdir,
        int(best_val_step),
        writer,
        load_descrs=False,
      )

  # Wait until computations are done before exiting.
  train_utils.barrier_across_hosts()

  # Return the train and eval summary after last step for regression testing.
  return train_state, train_summary, eval_summary