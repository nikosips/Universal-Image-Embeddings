from typing import Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

import flax.linen as nn



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