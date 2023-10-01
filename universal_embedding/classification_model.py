"""Class for all Universal Embedding project classification models."""

import functools
from typing import Dict, Optional, Tuple, Union

from flax.training import common_utils
import jax
import jax.numpy as jnp
import ml_collections
from scenic.model_lib.base_models import base_model
from scenic.model_lib.base_models import model_utils
from scenic.model_lib.base_models import multilabel_classification_model




def _transform_logits(
    logits,
    one_hot,
    loss_config,
):
  '''
  Transformation of the logits for different softmax margin losses.
  Transform type can be one of : [arcface, normface, cosface].
  '''

  if loss_config.transform_logits_type == "arcface":

    theta_yi = jax.lax.acos(logits * one_hot)
    
    transformed_logits = jax.lax.cos(
        theta_yi + loss_config.m
    ) * one_hot + logits * (1 - one_hot)

  elif loss_config.transform_logits_type == "cosface":  
    transformed_logits = (logits - loss_config.m) * one_hot + logits * (1 - one_hot)

  elif loss_config.transform_logits_type == "normface":  
    transformed_logits = logits
  
  transformed_logits *= loss_config.scale
  
  return transformed_logits



def classification_metrics_function(
    logits: jnp.array,
    batch: base_model.Batch,
    loss_config: ml_collections.ConfigDict,
    target_is_multihot: bool = False,
    axis_name: Union[str, Tuple[str, ...]] = 'batch',
    classifier = "separate",
) -> Dict[str, Tuple[float, int]]:
  """Calculates metrics for the multi-label classification task.

  Currently we assume each metric_fn has the API:
    ```metric_fn(logits, targets, weights)```
  and returns an array of shape [batch_size]. We also assume that to compute
  the aggregate metric, one should sum across all batches, then divide by the
  total samples seen. In this way we currently only support metrics of the 1/N
  sum f(inputs, targets). Note, the caller is responsible for dividing by
  the normalizer when computing the mean of each metric.

  Args:
   logits: Output of model in shape [batch, length, num_classes].
   batch: Batch of data that has 'label' and optionally 'batch_mask'.
   loss_config: the configuration for the loss.
   target_is_multihot: If the target is a multi-hot vector.
   axis_name: List of axes on which we run the pmsum.

  Returns:
    A dict of metrics, in which keys are metrics name and values are tuples of
    (metric, normalizer).
  """

  if target_is_multihot:
    multihot_target = batch['label']
  else:
    # This is to support running a multi-label classification model on
    # single-label classification tasks:    
    multihot_target = common_utils.onehot(batch['label'], logits.shape[-1])
    
  transformed_logits = _transform_logits(logits, multihot_target, loss_config)

  weights = batch.get('batch_mask')

  if classifier == "separate": 
    #because they will be fed to softmax, make the values that you dont want to use 
    #close to -inf, so they disappear and do not affect the loss
    transformed_logits = jnp.where(batch["domain_mask"],transformed_logits,jnp.finfo('float32').min)
  
  # This psum is required to correctly evaluate with multihost. Only host 0
  # will report the metrics, so we must aggregate across all hosts. The psum
  # will map an array of shape [n_global_devices, batch_size] -> [batch_size]
  # by summing across the devices dimension. The outer sum then sums across the
  # batch dim. The result is then we have summed across all samples in the
  # sharded batch.
  
  #logits for prec@1, transformed logits for loss
  evaluated_metrics = {
    
      'prec@1': model_utils.psum_metric_normalizer(
          (
              model_utils.weighted_top_one_correctly_classified(
                  logits, multihot_target, weights
              ),
              model_utils.num_examples(logits, multihot_target, weights),
          ),
          axis_name=axis_name,
      ),

      #the loss is masked if we use domain masks
      'loss': model_utils.psum_metric_normalizer(
          (
              model_utils.weighted_unnormalized_softmax_cross_entropy(
                  transformed_logits, multihot_target, weights
              ),
              model_utils.num_examples(
                  transformed_logits, multihot_target, weights
              ),
          ),
          axis_name=axis_name,
      ),
  }

  return evaluated_metrics



class UniversalEmbeddingClassificationModel(
    multilabel_classification_model.MultiLabelClassificationModel
):
  """Defines commonalities between all classification models.

  A model is class with three members: get_metrics_fn, loss_fn, & a flax_model.
  get_metrics_fn returns a callable function, metric_fn, that calculates the
  metrics and returns a dictionary. The metric function computes f(x_i, y_i) on
  a minibatch, it has API: ```metric_fn(logits, label, weights).``` The trainer
  will then aggregate and compute the mean across all samples evaluated. loss_fn
  is a function of API loss = loss_fn(logits, batch, model_params=None). This
  model class defines a softmax_cross_entropy_loss with weight decay, where the
  weight decay factor is determined by config.l2_decay_factor. flax_model is
  returned from the build_flax_model function. A typical usage pattern will be:
  ``` model_cls =
  model_lib.models.get_model_cls('fully_connected_classification') model =
  model_cls(config, dataset.meta_data) flax_model = model.build_flax_model
  dummy_input = jnp.zeros(input_shape, model_input_dtype) model_state, params =
  flax_model.init(

      rng, dummy_input, train=False).pop('params')
  ```
  And this is how to call the model:
  variables = {'params': params, **model_state}
  logits, new_model_state = flax_model.apply(variables, inputs, ...)
  ```
  """


  
  def get_metrics_fn(self, split: Optional[str] = None) -> base_model.MetricFn:
    """Returns a callable metric function for the model.

    Args:
      split: The split for which we calculate the metrics. It should be one of
        the ['train',  'validation', 'test'].
    Returns: A metric function with the following API: ```metrics_fn(logits,
      batch)```
    """
    del split  # For all splits, we return the same metric functions.
    return functools.partial(
        classification_metrics_function,
        target_is_multihot=self.dataset_meta_data.get(
            'target_is_onehot', False
        ),
        loss_config=self.config.loss,
    )



  def loss_function(
      self,
      logits: jnp.ndarray,
      batch: base_model.Batch,
      model_params: Optional[jnp.array] = None,
      classifier = "separate",
  ) -> float:
    """Returns the softmax loss.

    Args:
      logits: Output of model in shape [batch, length, num_classes].
      batch: Batch of data that has 'label' and optionally 'batch_mask'.
      model_params: Parameters of the model, for optionally applying
        regularization.

    Returns:
      Total loss.
    """

    # logits are cosine similarities at this point
    one_hot_targets = common_utils.onehot(batch['label'], logits.shape[-1])

    #transform based on type of softmax margin loss
    transformed_logits = _transform_logits(logits, one_hot_targets, self.config.loss)
    
    if classifier == "separate":

      transformed_masked_logits = jnp.where(batch["domain_mask"],transformed_logits,jnp.finfo('float32').min)
      sof_ce_loss = model_utils.weighted_softmax_cross_entropy(
          transformed_masked_logits,
          one_hot_targets,
          label_smoothing=self.config.get('label_smoothing'),
      )

      return sof_ce_loss

    elif classifier == "joint":

      sof_ce_loss = model_utils.weighted_softmax_cross_entropy(
          transformed_logits,
          one_hot_targets,
          label_smoothing=self.config.get('label_smoothing'),
      )

      return sof_ce_loss