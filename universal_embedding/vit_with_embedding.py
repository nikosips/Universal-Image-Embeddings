"""Vision Transformer with extra projection layer as the embedding,
and cosine classifier.
"""

from typing import Any, Callable, Iterable, Optional
from absl import logging


import ml_collections

import flax
import flax.linen as nn
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.model_lib.layers import nn_layers
from scenic.projects.baselines import vit
import scipy
from tensorflow.io import gfile

import jax

from universal_embedding import classification_model


Initializer = Callable[[jnp.ndarray, Iterable[int], jnp.dtype], jnp.ndarray]

JAX_PRECISION = "default"


class ViTWithEmbedding(vit.ViT):
  """Vision Transformer model which exposes the representation layer.

  Attributes:
  output_dim: output embedding dimension; if >0 will have projection layer, otherwise not
  """


  output_dim: int = -1
  dataset_meta_data: Any = None
  config: ml_collections.ConfigDict = None

  #We override the __call__ method of ViT class
  @nn.compact
  def __call__(
    self,
    x: jnp.ndarray, #batch of images
    domain: int,
    *,
    train: bool,
    init: bool = False,
    debug: bool = False,
    return_feats: bool = False,
    project_feats: bool = True,
  ):

    outputs = {}
    outputs['embeddings'] = {}

    if train or init:
      outputs['classifier'] = {}

    fh, fw = self.patches.size

    # Extracting patches and then embedding them to tokens is in fact a single convolution.
    x = nn.Conv(
        self.hidden_size,
        (fh, fw),
        strides=(fh, fw),
        padding='VALID',
        name='embedding',
        precision = jax.lax.Precision(JAX_PRECISION) #I put this here
    )(x)

    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    # If we want to add a class token, add it here.
    if self.classifier == 'token':
  
      cls = self.param('cls', nn.initializers.zeros, (1, 1, self.hidden_size), self.dtype)
      cls = jnp.tile(cls, [n, 1, 1])
      x = jnp.concatenate([cls, x], axis=1)

    x = vit.Encoder(
        mlp_dim=self.mlp_dim,
        num_layers=self.num_layers,
        num_heads=self.num_heads,
        positional_embedding=self.positional_embedding,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        stochastic_depth=self.stochastic_depth,
        dtype=self.dtype,
        name='Transformer',
    )(x, train=train)

    if self.classifier in ('token', '0'): #get the transformed cls token
      x = x[:, 0]
    elif self.classifier in ('gap', 'gmp', 'gsp'):
      fn = {'gap': jnp.mean, 'gmp': jnp.max, 'gsp': jnp.sum}[self.classifier]
      x = fn(x, axis=1)
    else:
      raise ValueError(f'Unknown classifier {self.classifier}')

    # l2 normalize the output
    x /= jnp.linalg.norm(x, ord=2, axis=1, keepdims=True)

    outputs['embeddings']['backbone_out'] = x

    if project_feats: # our projection layer for dim reduction.

      if self.output_dim > 0:
        x = nn.Dense(self.output_dim, name='projection')(x)
      else:
        x = nn_layers.IdentityLayer(name='projection')(x)
      # l2 norm the embeddings again
      x /= jnp.linalg.norm(x, ord=2, axis=1, keepdims=True)

      outputs['embeddings']['projected'] = x #TODO: use of project_feats flag is not needed anymore since they are two different named embeddings now


    if not return_feats: # pass through classification layer #(can be replaced with if train or init?)
      
      current_dataset=self.dataset_meta_data["dataset_name"].split(",")[domain]
      
      #add classifier to new file
      if self.config.classifier=="separate":
        classifier_domain=domain
        classifier_num_classes=self.dataset_meta_data['classes_per_dataset'][current_dataset]
      elif self.config.classifier=="joint":
        classifier_domain=0 #1 "domain" for all that will contain the sum of the classes
        classifier_num_classes=self.num_classes#sum classes

      x = nn.Dense(
        classifier_num_classes,
        use_bias=False,
        kernel_init=nn.initializers.lecun_uniform(),
        name=f'output_projection_{classifier_domain}',
      )(x)
      weights_norms = jnp.linalg.norm(
        self.variables['params'][f'output_projection_{classifier_domain}']['kernel'], axis=0
      )  # norms of class prototypes
      x /= weights_norms

      outputs['classifier']['logits'] = x

    return outputs



class ViTWithEmbeddingClassificationModel(
    classification_model.UniversalEmbeddingClassificationModel
):
  """Vision Transformer model for classification task."""

  #overriding the method of scenic
  def build_flax_model(self) -> nn.Module:
    dtype_str = self.config.get('model_dtype_str', 'float32')

    if dtype_str != 'float32':
      raise ValueError(
          '`dtype` argument is not propagated properly '
          'in the current implmentation, so only '
          '`float32` is supported for now.'
      )
  
    return ViTWithEmbedding(
        
        num_classes=self.dataset_meta_data['num_classes'], #total classes
        dataset_meta_data=self.dataset_meta_data,
        config=self.config,
        
        mlp_dim=self.config.model.mlp_dim,
        num_layers=self.config.model.num_layers,
        num_heads=self.config.model.num_heads,
        positional_embedding=self.config.model.get(
            'positional_embedding', 'learned_1d'
        ),
        representation_size=self.config.model.representation_size,
        patches=self.config.model.patches,
        hidden_size=self.config.model.hidden_size,
        classifier=self.config.model.classifier,
        dropout_rate=self.config.model.get('dropout_rate'),
        attention_dropout_rate=self.config.model.get('attention_dropout_rate'),
        stochastic_depth=self.config.model.get('stochastic_depth', 0.0),
        dtype=getattr(jnp, dtype_str),

        output_dim=self.config.model.output_dim,
    
    )


  def default_flax_model_config(self) -> ml_collections.ConfigDict:

    return ml_collections.ConfigDict(
      {
        'model': dict(
          num_heads=2,
          num_layers=1,
          representation_size=16,
          mlp_dim=32,
          dropout_rate=0.0,
          attention_dropout_rate=0.0,
          hidden_size=16,
          patches={'size': (4, 4)},
          classifier='gap',
          data_dtype_str='float32',
        )
      }
    )

  def init_from_train_state(
      self,
      train_state: Any,
      restored_train_state: Any,
      restored_model_cfg: ml_collections.ConfigDict,
  ) -> Any:
    """Updates the train_state with data from restored_train_state.

    This function is writen to be used for 'fine-tuning' experiments. Here, we
    do some surgery to support larger resolutions (longer sequence length) in
    the transformer block, with respect to the learned pos-embeddings.

    Args:
      train_state: A raw TrainState for the model.
      restored_train_state: A TrainState that is loaded with parameters/state of
        a pretrained model.
      restored_model_cfg: Configuration of the model from which the
        restored_train_state come from. Usually used for some asserts.

    Returns:
      Updated train_state.
    """
    return vit.init_vit_from_train_state(
        train_state, restored_train_state, self.config, restored_model_cfg
    )


  def load_augreg_params(
      self,
      train_state: Any,
      params_path: str,
      model_cfg: ml_collections.ConfigDict,
  ) -> Any:
    """Loads parameters from an AugReg checkpoint.

    See
    https://github.com/google-research/vision_transformer/
    and
    https://arxiv.org/abs/2106.10270
    for more information about these pre-trained models.

    Args:
      train_state: A raw TrainState for the model.
      params_path: Path to an Augreg checkpoint. The model config is read from
        the filename (e.g. a B/32 model starts with "B_32-").
      model_cfg: Configuration of the model. Usually used for some asserts.

    Returns:
      Updated train_state with params replaced with the ones read from the
      AugReg checkpoint.
    """

    restored_model_cfg = _get_augreg_cfg(params_path,model_cfg)
    assert tuple(restored_model_cfg.patches.size) == tuple(
        model_cfg.patches.size
    )
    assert restored_model_cfg.hidden_size == model_cfg.hidden_size
    assert restored_model_cfg.mlp_dim == model_cfg.mlp_dim
    assert restored_model_cfg.num_layers == model_cfg.num_layers
    assert restored_model_cfg.num_heads == model_cfg.num_heads
    assert restored_model_cfg.classifier == model_cfg.classifier

    flattened = np.load(gfile.GFile(params_path, 'rb'))
    restored_params = flax.traverse_util.unflatten_dict(
        {tuple(k.split('/')): v for k, v in flattened.items()}
    )
    restored_params['output_projection'] = restored_params.pop('head')

    params = flax.core.unfreeze(train_state.params)
    _merge_params(params, restored_params, model_cfg, restored_model_cfg)
    return train_state.replace(params=flax.core.freeze(params))


def _get_augreg_cfg(params_path,model_cfg):

  model = params_path.split('/')[-1].split('-')[0]

  return ml_collections.ConfigDict(
      dict(
          num_classes=0,
          mlp_dim=model_cfg.mlp_dim,
          num_layers=model_cfg.num_layers,
          num_heads=model_cfg.num_heads,
          hidden_size=model_cfg.hidden_size,
          classifier=model_cfg.classifier,
          patches=dict(size=tuple(model_cfg.patches.size)),
          dropout_rate=model_cfg.dropout_rate,
          attention_dropout_rate=model_cfg.attention_dropout_rate,
      )
  )


def _merge_params(params, restored_params, model_cfg, restored_model_cfg):

  """Merges `restored_params` into `params`."""
  # Start moving parameters, one-by-one and apply changes if needed.
  for m_key, m_params in restored_params.items():
  
    if m_key == 'output_projection':
      # For the classifier head, we use a the randomly initialized params and
      # ignore the the one from pretrained model.
      pass

    elif m_key == 'pre_logits':
      if model_cfg.model.representation_size is None:
        # We don't have representation_size in the new model, so let's ignore
        #   it from the pretained model, in case it has it.
        # Note, removing the key from the dictionary is necessary to prevent
        #   obscure errors from the Flax optimizer.
        params.pop(m_key, None)
      else:
        assert restored_model_cfg.model.representation_size
        params[m_key] = m_params

    elif m_key == 'Transformer':
      for tm_key, tm_params in m_params.items():
        if tm_key == 'posembed_input':  # Might need resolution change.
          posemb = params[m_key]['posembed_input']['pos_embedding']
          restored_posemb = m_params['posembed_input']['pos_embedding']

          if restored_posemb.shape != posemb.shape:
            # Rescale the grid of pos, embeddings: param shape is (1, N, d).
            logging.info(
                'Resized variant: %s to %s', restored_posemb.shape, posemb.shape
            )
            ntok = posemb.shape[1]
            if restored_model_cfg.model.classifier == 'token':
              # The first token is the CLS token.
              restored_posemb_grid = restored_posemb[0, 1:]
              if model_cfg.model.classifier == 'token':
                # CLS token in restored model and in target.
                cls_tok = restored_posemb[:, :1]
                ntok -= 1
              else:
                # CLS token in restored model, but not target.
                cls_tok = restored_posemb[:, :0]
            else:
              restored_posemb_grid = restored_posemb[0]
              if model_cfg.model.classifier == 'token':
                # CLS token in target, but not restored model.
                cls_tok = posemb[:, :1]
                ntok -= 1
              else:
                # CLS token not in target or restored model.
                cls_tok = restored_posemb[:, :0]

            restored_gs = int(np.sqrt(len(restored_posemb_grid)))
            gs = int(np.sqrt(ntok))
            if restored_gs != gs:  # We need resolution change.
              logging.info('Grid-size from %s to %s.', restored_gs, gs)
              restored_posemb_grid = restored_posemb_grid.reshape(
                  restored_gs, restored_gs, -1
              )
              zoom = (gs / restored_gs, gs / restored_gs, 1)
              restored_posemb_grid = scipy.ndimage.zoom(
                  restored_posemb_grid, zoom, order=1
              )
            # Attach the CLS token again.
            restored_posemb_grid = restored_posemb_grid.reshape(1, gs * gs, -1)
            restored_posemb = jnp.array(
                np.concatenate([cls_tok, restored_posemb_grid], axis=1)
            )

          params[m_key][tm_key]['pos_embedding'] = restored_posemb

        # Other parameters of the Transformer encoder if they are in the target.
        elif tm_key in params[m_key]:
          # Needs the following changes:
          # MultiHeadDotProductAttention_1 to MultiHeadDotProductAttention_0.
          # LayerNorm_2 to LayerNorm_1
          # MlpBlock_3 to MlpBlock_0
          tm_params_updated = {}
          for key, val in tm_params.items():
            new_key = (
                key.replace(
                    'MultiHeadDotProductAttention_1',
                    'MultiHeadDotProductAttention_0',
                )
                .replace('LayerNorm_2', 'LayerNorm_1')
                .replace('MlpBlock_3', 'MlpBlock_0')
            )
            tm_params_updated[new_key] = val
          params[m_key][tm_key] = tm_params_updated
        else:
          logging.info(
              "Ignoring %s. In restored model's Transformer,but not in target",
              m_key,
          )

    elif m_key in params:
      # Use the rest if they are in the pretrained model.
      params[m_key] = m_params

    else:
      logging.info('Ignoring %s. In restored model, but not in target', m_key)