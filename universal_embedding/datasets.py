"""Data generators for the universal embedding datasets."""

import collections
import functools
import os
from typing import Optional, Union, List

from absl import logging
from flax import jax_utils
import jax
import jax.numpy as jnp
import ml_collections
from scenic.dataset_lib import dataset_utils
import tensorflow as tf
from tensorflow.io import gfile

import json

import numpy as np
from collections import OrderedDict


PRNGKey = jnp.ndarray

IMAGE_RESIZE = 256
IMAGE_SIZE = 224

MEAN_RGB = [0.5,0.5,0.5]
STDDEV_RGB = [0.5,0.5,0.5]



UniversalEmbeddingTrainingDataset = collections.namedtuple(
    # Each instance of the Dataset.
    'TrainingDataset',
    [
        'train_iter',
        'meta_data',
    ],
)

UniversalEmbeddingKnnEvalDataset = collections.namedtuple(
    # Each instance of the Dataset.
    'KnnDataset',
    [
        'knn_info',
        'meta_data',
    ],
)


#if any of the below 2 dicts' order changes, the other one has to change as well
DOMAIN_LABELS = {
    'cars': 0,
    'sop': 1,
    'inshop': 2,
    'inat': 3,
    'met': 4,
    'gldv2': 5,
    'food2k': 6,
    'rp2k': 7,
}


invert_DOMAIN_LABELS = {
    0: 'cars',
    1: 'sop',
    2: 'inshop',
    3: 'inat',
    4: 'met',
    5: 'gldv2',
    6: 'food2k',
    7: 'rp2k',
}


DATASET_INFO = {
    'cars': {
        'domain': 0,
        'train_files': 'cars/train/cars.train.tfrecord',
        'test_files': 'cars/test/cars.test.tfrecord',
        'val_files': 'cars/val/cars.val.tfrecord',

        'num_train_classes': 78,
        'num_train_examples': 6346,
        'num_test_examples': 8131,
        'num_val_examples': 1708,
        'knn': {
            'train_knn': {
                'query': 'train',
                'index': 'train',
            },
            'val_knn': {
                'query': 'val',
                'index': 'val',
            },
            'test_knn': {
                'query': 'test',
                'index': 'test',
            },
        },
    },
    'sop': {
        'domain': 1,
        'train_files': 'sop/train/sop.train.tfrecord',
        'test_files': 'sop/test/sop.test.tfrecord',
        'val_files': 'sop/val/sop.val.tfrecord',

        'num_train_classes': 9054,
        'num_train_examples': 48942,
        'num_test_examples': 60502,
        'num_val_examples': 10609,
        'knn': {
            'train_knn': {
                'query': 'train',
                'index': 'train',
            },
            'val_knn': {
                'query': 'val',
                'index': 'val',
            },
            'test_knn': {
                'query': 'test',
                'index': 'test',
            },
        },
    },

    'inshop': {
        'domain': 2,
        'train_files': 'inshop/train/inshop.train.tfrecord',
        'test_files': 'inshop/test/inshop.test.tfrecord',
        'val_files': 'inshop/val/inshop.val.tfrecord',
        'index_files': 'inshop/index/inshop.index.tfrecord',  # size 12612

        'num_train_classes': 3198,
        'num_train_examples': 20897,
        'num_test_examples': 14218,
        'num_val_examples': 4982,
        'knn': {
            'train_knn': {
                'query': 'train',
                'index': 'train',
            },
            'val_knn': {
                'query': 'val',
                'index': 'val',
            },
            'test_knn': {
                'query': 'test',
                'index': 'index',
            },
        },
    },

    'inat': {
        'domain': 3,
        'train_files': 'inat/train/inat.train.tfrecord',
        'test_files': 'inat/test/inat.test.tfrecord',
        'val_files': 'inat/val/inat.val.tfrecord',

        'num_train_classes': 4552,
        'num_train_examples': 273929,
        'num_test_examples': 136093,
        'num_val_examples': 51917,
        'knn': {
            'train_knn': {
                'query': 'train',
                'index': 'train',
            },
            'val_knn': {
                'query': 'val',
                'index': 'val',
            },
            'test_knn': {
                'query': 'test',
                'index': 'test',
            },
        },
    },

    'met': {
        'domain': 4,
        'train_files': 'met/train/met.train.tfrecord',
        'small_train_files': 'met/small_train/met.small_train.tfrecord',  # size 38307
        'test_files': 'met/test/met.test.tfrecord',
        'val_files': 'met/val/met.val.tfrecord',

        'index_files': 'met/index/met.train.tfrecord', #same as train set
        'small_index_files': 'met/small_index/met.small_train.tfrecord', #same as small train set

        'num_train_classes': 224408,
        'num_train_examples': 397121,
        'num_test_examples': 1003,
        'num_val_examples': 129,
        'knn': {
            'train_knn': {
                'query': 'train',
                'index': 'train',
            },
            'val_knn': {
                'query': 'val',
                'index': 'small_index', 
            },
            'test_knn': {
                'query': 'test',
                'index': 'index',
            },
        },
    },

    'gldv2': {
        'domain': 5,
        'train_files': 'gldv2/train/gldv2.train_clean.tfrecord',        
        'test_files': 'gldv2/test/gldv2.test.tfrecord',
        'val_files': 'gldv2/val/gldv2.val_train.tfrecord',
        'index_files': 'gldv2/index/gldv2.index.tfrecord',  # size 761757

        'num_train_classes': 73182,
        'num_train_examples': 1422914,
        'num_test_examples': 1129, 
        'num_val_examples': 157556,
        'knn': {
            'train_knn': {
                'query': 'train',
                'index': 'train',
            },
            'val_knn': {
                'query': 'val',
                'index': 'val',
            },
            'test_knn': {
                'query': 'test',
                'index': 'index',
            },
        },
    },

    'food2k': {
        'domain': 6,
        'train_files': 'food2k/train/food2k.train.tfrecord',
        'test_files': 'food2k/test/food2k.test.tfrecord',
        'val_files': 'food2k/val/food2k.val.tfrecord',


        'num_train_classes': 900,
        'num_train_examples': 472349,
        'num_test_examples': 9979,
        'num_val_examples': 49323,
        'knn': {
            'train_knn': {
                'query': 'train',
                'index': 'train',
            },
            'val_knn': {
                'query': 'val',
                'index': 'val',
            },
            'test_knn': {
                'query': 'test',
                'index': 'test',
            },
        },
    },

    'rp2k': {
        'domain': 7,
        'train_files': 'rp2k/train/rp2k.train.tfrecord',
        'test_files': 'rp2k/test/rp2k.test.tfrecord',
        'val_files': 'rp2k/val/rp2k.val.tfrecord',
        

        'num_train_classes': 1074,
        'num_train_examples': 188724,
        'num_test_examples': 10931,
        'num_val_examples': 17185,
        'knn': {
            'train_knn': {
                'query': 'train',
                'index': 'train',
            },
            'val_knn': {
                'query': 'val',
                'index': 'val',
            },
            'test_knn': {
                'query': 'test',
                'index': 'test',
            },
        },
    },
}


def _normalize_image(
    image,
    normalization_statistics,
):

  if normalization_statistics is None:

      image /= tf.constant(255, shape=[1, 1, 3], dtype=image.dtype)
      image -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
      image /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)

  else:

      image /= tf.constant(255, shape=[1, 1, 3], dtype=image.dtype)
      image -= tf.constant(normalization_statistics["MEAN_RGB"], shape=[1, 1, 3], dtype=image.dtype)
      image /= tf.constant(normalization_statistics["STDDEV_RGB"], shape=[1, 1, 3], dtype=image.dtype)


  return image


def _process_train_split(image):

  # Resize to 256x256
  image = _resize(image, IMAGE_RESIZE)
  # Random crop to 224x224
  image = tf.image.random_crop(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
  # Random flip
  image = tf.image.random_flip_left_right(image)
  return image


def _process_test_split(image): 
  
  # Resize the small edge to 224.
  image, new_size = _resize_smaller_edge(image, IMAGE_SIZE)
  # Central crop to 224x224.
  h, w = new_size
  if h > w:
    image = tf.image.crop_to_bounding_box(image, (h - w) // 2, 0, w, w)
  else:
    image = tf.image.crop_to_bounding_box(image, 0, (w - h) // 2, h, h)

  return image


def _resize(image, image_size):
  
  """
  Resizes the image.

  Args:
    image: Tensor; Input image.
    image_size: int; Image size.

  Returns:
    Resized image.
  """
  
  return tf.image.resize(
      image, 
      [image_size, image_size], 
      method=tf.image.ResizeMethod.BILINEAR,
  )


def _resize_smaller_edge(
  image, 
  image_size,
):
  
  """Resizes the smaller edge to the desired size and keeps the aspect ratio."""
  
  shape = tf.shape(image)
  height, width = shape[0], shape[1]
  if height <= width:
    # Resize to [224, width / height * 224]
    new_height = image_size
    new_width = tf.cast((width / height) * image_size, tf.int32)
  else:
    # Resize to [height / width *224, 224]
    new_width = image_size
    new_height = tf.cast((height / width) * image_size, tf.int32)

  return tf.image.resize(
      image, 
      [new_height, new_width], 
      method=tf.image.ResizeMethod.BILINEAR,
  ),(new_height, new_width)



def preprocess_example(
  example, 
  split,
  total_classes,
  domain, 
  augment=False, 
  dtype=tf.float32, 
  label_offset=0,
  domain_mask_range = None,
  domain_idx = -1,
  normalization_statistics = None,
):
  """Preprocesses the given image.

  Args:
    example: The proto of the current example.
    split: str; One of 'train' or 'test'.
    domain: int; the domain of the dataset.
    augment: whether to augment the image.
    dtype: Tensorflow data type; Data type of the image.
    label_offset: int; The offset of label id.

  Returns:
    A preprocessed image `Tensor`.
  """
  
  features = tf.io.parse_single_example(
      example,
      features={
        'image_bytes': tf.io.FixedLenFeature([], tf.string),
        'key': tf.io.FixedLenFeature([], tf.string),
        'class_id': tf.io.FixedLenFeature([], tf.int64),
      },
  )
  
  image = tf.io.decode_jpeg(features['image_bytes'], channels=3)

  if split == 'train' and augment:
    image = _process_train_split(image)
  else:
    image = _process_test_split(image)

  image = _normalize_image(image,normalization_statistics)
  image = tf.image.convert_image_dtype(image, dtype=dtype)
  
  domain_mask = np.full((total_classes),False)
  domain_mask[domain_mask_range[0]:domain_mask_range[1]] = True

  return {
    'inputs': image,
    'label': features['class_id'] + label_offset,
    'domain': domain,
    'domain_mask': domain_mask,
    'domain_idx': domain_idx,
  }


def preprocess_example_eval(
  example, 
  split,
  total_classes,
  domain, 
  augment=False, 
  dtype=tf.float32, 
  domain_mask_range = None,
  domain_idx = -1,
  normalization_statistics = None,
):
  """Preprocesses the given image.

  Args:
    example: The proto of the current example.
    split: str; One of 'train' or 'test'.
    domain: int; the domain of the dataset.
    augment: whether to augment the image.
    dtype: Tensorflow data type; Data type of the image.

  Returns:
    A preprocessed image `Tensor`.
  """
  
  features = tf.io.parse_single_example(
      example,
      features={
        'image_bytes': tf.io.FixedLenFeature([], tf.string),
        'key': tf.io.FixedLenFeature([], tf.string),
      },
  )
  
  image = tf.io.decode_jpeg(features['image_bytes'], channels=3)

  if split == 'train' and augment:
    image = _process_train_split(image)
  else:
    image = _process_test_split(image)

  image = _normalize_image(image,normalization_statistics)
  image = tf.image.convert_image_dtype(image, dtype=dtype)

  domain_mask = np.full((total_classes),False)
  domain_mask[domain_mask_range[0]:domain_mask_range[1]] = True

  return {
    'inputs': image,
    'domain': domain,
    'domain_mask': domain_mask,
    'domain_idx': domain_idx,
  }



def load_tfrecords(
    base_dir, 
    dataset_name, 
    split,
    total_classes,
    augment=False, 
    parallel_reads=4, 
    label_offset=0,
    domain_mask_range = None,
    domain_idx = -1,
    **dataset_kwargs,
):
  """Loads the tfds.

  Args:
    dataset_name: str; name of the dataset.
    split: str; One of 'train', 'val', 'test' or 'index.
    augment: whether to augment the images.
    parallel_reads: int; Number of parallel readers (set to 1 for determinism).
    label_offset: int; The offset of label id.

  Returns:
    tf.data.Datasets for training, testing and validation. if
    n_validation_shards is 0, the validation dataset will be None.
  """
  dataset_info = DATASET_INFO[dataset_name]
  file_name = split + '_files'
  
  path = os.path.join(base_dir, dataset_info[file_name])

  if not dataset_kwargs['knn']:

    files = tf.data.Dataset.list_files(path + "*") #use this when you want to shuffle

    files = files.shuffle(1000)

    data = files.interleave(
        tf.data.TFRecordDataset,
        cycle_length=1 if split != 'train' or not augment else parallel_reads,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

  else:
    
    files = tf.data.Dataset.list_files(path + "*",shuffle = False)
    
    #files are not shuffled until here
    data = tf.data.TFRecordDataset(files)


  #a dict of mean and std for image normalization
  normalization_statistics = dataset_kwargs["config"].get("normalization_statistics",None)


  def _preprocess_example(example):

    if dataset_kwargs['knn']:
      
      return preprocess_example_eval(
        example,
        split,
        total_classes,
        domain=DATASET_INFO[dataset_name]['domain'],
        domain_idx=domain_idx,
        augment=augment,
        domain_mask_range = domain_mask_range,
        normalization_statistics = normalization_statistics,
      )

    else:

      return preprocess_example(
        example,
        split,
        total_classes,
        domain=DATASET_INFO[dataset_name]['domain'],
        domain_idx=domain_idx,
        augment=augment,
        label_offset=label_offset,
        domain_mask_range = domain_mask_range,
        normalization_statistics = normalization_statistics,
      )


  if not dataset_kwargs['knn']:

    data = data.map(
        _preprocess_example,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

  else:

    data = data.map(
          _preprocess_example,
          num_parallel_calls=1,
      )


  return data



def build_dataset_new(
  dataset_fn,
  batch_size=None,
  shuffle_buffer_size=256,
  seed=None,
  repeat=False,
  sampling=None,
  knn = False,
  **dataset_kwargs,
):
  """Dataset builder that takes care of strategy, batching and shuffling.

  Args:
    dataset_fn: function; A function that loads the dataset.
    batch_size: int; Size of the batch.
    shuffle_buffer_size: int; Size of the buffer for used for shuffling.
    seed: int; Random seed used for shuffling.
    repeat: bool; Whether to repeat the dataset.
    sampling: str; the sampling option for multiple datasets
    **dataset_kwargs: dict; Arguments passed to TFDS.

  Returns:
    Dataset.
  """

  dataset_kwargs['knn'] = knn

  def _shuffle_batch_prefetch(dataset, replica_batch_size, split):
    
    if split == 'train' and repeat:
    
      dataset = dataset.shuffle(
          shuffle_buffer_size, seed=seed, reshuffle_each_iteration=True,
      )
      dataset = dataset.batch(replica_batch_size, drop_remainder=True)
    
      #shuffle the batches again
      batch_shuffle_buffer_size = 16

      dataset = dataset.shuffle(
          batch_shuffle_buffer_size, seed=seed, reshuffle_each_iteration=True,
      )

    else:

      #knn case
      dataset = dataset.batch(replica_batch_size, drop_remainder=False)

    options = tf.data.Options()
    options.experimental_optimization.parallel_batch = True
    dataset = dataset.with_options(options)
    
    return dataset.prefetch(tf.data.experimental.AUTOTUNE)


  def _dataset_fn(input_context=None):
    """Dataset function."""

    replica_batch_size = batch_size

    if input_context:
      replica_batch_size = input_context.get_per_replica_batch_size(batch_size)

    #dataset_fn here is "build_universal_embedding_dataset_new"
    ds_dict = dataset_fn(**dataset_kwargs)

    split = dataset_kwargs.get('split')
    
    if split == 'train' and repeat:

      for ds_name,ds in ds_dict.items():
        ds = ds.repeat()
        ds_dict[ds_name] = _shuffle_batch_prefetch(ds, replica_batch_size, split)

      return ds_dict
    
    else:

      #case of knn dataset (only one domain at a a time)
      assert len(list(ds_dict.keys())) == 1
      domain_name = list(ds_dict.keys())[0]

      return _shuffle_batch_prefetch(ds_dict[domain_name], replica_batch_size, split)

  return _dataset_fn()



def build_ds_iter(
  ds, 
  maybe_pad_batches, 
  shard_batches, 
  prefetch_buffer_size,
):
    
  
  ds_iter = iter(ds)
  ds_iter = map(dataset_utils.tf_to_numpy, ds_iter)
  ds_iter = map(maybe_pad_batches, ds_iter)
  ds_iter = map(shard_batches, ds_iter)
  ds_iter = jax_utils.prefetch_to_device(ds_iter, prefetch_buffer_size)
  
  return ds_iter



def build_universal_embedding_dataset_new(
  base_dir,
  dataset_names, 
  split, 
  augment=False,
  **dataset_kwargs,
):
  """dataset_fn called by data.build_dataset(**kwargs)."""

  total_classes = 0

  for i in range(len(dataset_names)):
    total_classes += DATASET_INFO[dataset_names[i]]['num_train_classes']

  ds_dict = {}
  offset = 0

  for i in range(len(dataset_names)):

    new_ds = load_tfrecords(
        base_dir,
        dataset_names[i],
        split,
        total_classes,
        parallel_reads=4,
        augment=augment,
        label_offset=offset,
        domain_mask_range=(offset,offset + DATASET_INFO[dataset_names[i]]['num_train_classes']),
        domain_idx=i,
        **dataset_kwargs,
    )

    offset += DATASET_INFO[dataset_names[i]]['num_train_classes']
    ds_dict[dataset_names[i]]=new_ds
      
  return ds_dict



def get_training_dataset_new(
  config: ml_collections.ConfigDict,
  num_local_shards: Optional[int] = None,
  prefetch_buffer_size: Optional[int] = 2,
  dataset_configs: Optional[ml_collections.ConfigDict] = None,
):
  """Returns generators for the universal embedding train, validation, and test sets.

  Args:
    config: The configuration of the experiment.
    data_rng: Random number generator key to use for the dataset.
    num_local_shards: Number of shards for each batch. So (bs, ...) becomes
      (num_local_shards, bs//num_local_shards, ...). If not specified, it will
      be number of local devices.
    prefetch_buffer_size: int; Buffer size for the device prefetch.
    dataset_configs: Configuration of the dataset, if not reading directly from
      the config.

  Returns:
    A dataset_utils.Dataset() which includes a train_iter, a valid_iter,
      a test_iter, and a dict of meta_data.
  """
  device_count = jax.device_count()
  logging.info('device_count: %d', device_count)
  logging.info('num_hosts : %d', jax.process_count())
  logging.info('host_id : %d', jax.process_index())

  base_dir = config.train_dataset_dir

  dataset_name = config.dataset_name
  dataset_names = dataset_name.split(',')
  batch_size = config.batch_size

  if batch_size % device_count > 0:
    raise ValueError(
        f'Batch size ({batch_size}) must be divisible by the '
        f'number of devices ({device_count})'
    )

  local_batch_size = batch_size // jax.process_count()
  logging.info('local_batch_size : %d', local_batch_size)

  eval_batch_size = config.get('eval_batch_size', batch_size)
  local_eval_batch_size = eval_batch_size // jax.process_count()
  logging.info('local_eval_batch_size : %d', local_eval_batch_size)

  shuffle_seed = config.get('shuffle_seed', None)

  dataset_configs = dataset_configs or config.get('dataset_configs', {})
  num_local_shards = num_local_shards or jax.local_device_count()
  logging.info('local_eval_batch_size : %d', local_eval_batch_size)

  # use different seed for each host
  if shuffle_seed is None:
    local_seed = None
  else:
    data_seed = 0
    local_seed = data_seed + jax.process_index()

  maybe_pad_batches_train = functools.partial(
      dataset_utils.maybe_pad_batch,
      train=True,
      batch_size=local_batch_size,
  )

  shard_batches = functools.partial(
      dataset_utils.shard, n_devices=num_local_shards
  )

  train_ds_dict = build_dataset_new(
    dataset_fn=build_universal_embedding_dataset_new,
    dataset_names=dataset_names,
    split='train',
    batch_size=local_batch_size,
    seed=local_seed,
    augment=True,
    repeat=True,
    sampling=dataset_configs.get('sampling', None),
    base_dir = base_dir,
    config = config,
  )

  train_iter_dict = {}

  for ds_name,ds in train_ds_dict.items():
    train_iter_dict[ds_name] = build_ds_iter(
      train_ds_dict[ds_name], maybe_pad_batches_train, shard_batches, prefetch_buffer_size
  )

  input_shape = (
    -1,
    IMAGE_SIZE,
    IMAGE_SIZE,
    3,
  )

  num_train_examples, num_test_examples, num_val_examples, num_classes = (
    0,
    0,
    0,
    0,
  )

  dataset_samples = OrderedDict()

  for name in dataset_names:
    num_train_examples += DATASET_INFO[name]['num_train_examples']
    num_test_examples += DATASET_INFO[name]['num_test_examples']
    num_val_examples += DATASET_INFO[name]['num_val_examples']
    num_classes += DATASET_INFO[name]['num_train_classes']
    dataset_samples[name] = DATASET_INFO[name]['num_train_examples']


  domain_indices = [DATASET_INFO[dat_name]["domain"] for dat_name in dataset_names]

  meta_data = {
    'dataset_name': dataset_name,
    'domain_indices': domain_indices,
    'num_classes': num_classes,
    'input_shape': input_shape,
    'num_train_examples': num_train_examples,
    'num_test_examples': num_test_examples,
    'num_val_examples': num_val_examples,
    'input_dtype': getattr(jnp, config.data_dtype_str),
    'target_is_onehot': False,
    'dataset_samples': dataset_samples,
  }

  return UniversalEmbeddingTrainingDataset(
      train_iter_dict,
      meta_data,
  )



def dataset_lookup_key(dataset_name, split):
  return dataset_name + ':' + split



def get_knn_eval_datasets(
    config,
    base_dir,
    dataset_names: Union[List[str], str],
    eval_batch_size: int,
    prefetch_buffer_size: Optional[int] = 2,
):
  """Returns generators for the universal embedding train, validation, and test sets.

  Args:
    dataset_names: a lsit of dataset names.
    eval_batch_size: The eval batch size.
    prefetch_buffer_size: int; Buffer size for the device prefetch.

  Returns:
    A dataset_utils.Dataset() which includes a train_iter, a valid_iter,
      a test_iter, and a dict of meta_data.
  """

  base_dir = config.eval_dataset_dir

  device_count = jax.device_count()
  logging.info('device_count: %d', device_count)
  logging.info('num_hosts : %d', jax.process_count())
  logging.info('host_id : %d', jax.process_index())

  local_eval_batch_size = eval_batch_size // jax.process_count()

  logging.info('local_eval_batch_size : %d', local_eval_batch_size)

  num_local_shards = jax.local_device_count()

  maybe_pad_batches_eval = functools.partial(
    dataset_utils.maybe_pad_batch,
    train=False,
    batch_size=local_eval_batch_size,
  )

  shard_batches = functools.partial(
    dataset_utils.shard, 
    n_devices=num_local_shards,
  )

  if isinstance(dataset_names, str):
    dataset_names = dataset_names.split(',')

  knn_info, knn_setup, size_info = {}, {}, {}

  knn_info['json_data'] = {}


  for dataset_name in dataset_names:
  
    knn_splits = set()
  
    for val in DATASET_INFO[dataset_name]['knn'].values():
      knn_splits.add(val['query'])
      knn_splits.add(val['index'])      


    for split in knn_splits:

      split_knn_ds = build_dataset_new(
        dataset_fn=build_universal_embedding_dataset_new,
        dataset_names=[dataset_name],
        split=split,
        batch_size=local_eval_batch_size,
        base_dir = base_dir,
        config = config,
        knn = True,
      )

      split_knn_iter = build_ds_iter(
        split_knn_ds,
        maybe_pad_batches_eval,
        shard_batches,
        prefetch_buffer_size,
      )
      
      knn_info[dataset_lookup_key(dataset_name, split)] = split_knn_iter

      json_data = {}

      #load the labels and the rest of the data for every sample here from the .json files
      json_path = os.path.join(config.info_files_dir,dataset_name)
      json_path = os.path.join(json_path,f"{split}.json")

      with gfile.GFile(json_path, 'rb') as f:
        split_info = json.load(f)

      paths = []
      labels = []
      for sample in split_info:
        paths.append(sample["path"])
        labels.append(sample["class_id"])

      json_data["paths"] = paths
      json_data["labels"] = labels
      json_data["domains"] = [DOMAIN_LABELS[dataset_name] for _ in range(len(labels))]

      knn_info['json_data'][dataset_lookup_key(dataset_name, split)] = json_data

    knn_setup[dataset_name] = DATASET_INFO[dataset_name]['knn']
  
    size_info[dataset_name] = {
      'num_train_examples': DATASET_INFO[dataset_name]['num_train_examples'],
      'num_test_examples': DATASET_INFO[dataset_name]['num_test_examples'],
      'num_val_examples': DATASET_INFO[dataset_name]['num_val_examples'],
    }

  knn_info['knn_setup'] = knn_setup
  
  meta_data = {
    'dataset_names': ','.join(dataset_names),
    'top_k': int(config.top_k),
    'size_info': size_info,
  }


  return UniversalEmbeddingKnnEvalDataset(
    knn_info,
    meta_data,
  )