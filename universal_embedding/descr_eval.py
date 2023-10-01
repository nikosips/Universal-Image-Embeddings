"""Script for Knn evalulation without descriptor extraction."""

from clu import metric_writers

import jax
import jax.numpy as jnp
import ml_collections
from scenic import app
from scenic.train_lib import train_utils

import sys


if sys.version_info.major == 3 and sys.version_info.minor >= 10:

  from collections.abc import MutableMapping
else:
  from collections import MutableMapping

from universal_embedding import knn_utils as knn_utils



def knn_evaluate(
  rng: jnp.ndarray,
  config: ml_collections.ConfigDict,
  workdir: str,
  writer: metric_writers.MetricWriter,
) -> None:

  
  lead_host = jax.process_index() == 0
  
  knn_eval_batch_size = config.get('knn_eval_batch_size') or config.batch_size
  representation_fn_knn = None

  knn_evaluator = knn_utils.KNNEvaluator(
    config,
    representation_fn_knn,
    knn_eval_batch_size,
    False,
  )

  train_state = None

  knn_utils.knn_single(
    knn_evaluator,
    train_state,
    config,
    writer,
    workdir,
  )

  train_utils.barrier_across_hosts()



if __name__ == '__main__':
  app.run(main=knn_evaluate)