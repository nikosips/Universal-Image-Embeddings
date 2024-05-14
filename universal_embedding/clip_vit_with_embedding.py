import functools
from typing import Optional, Mapping, Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.model_lib.layers import nn_layers
from universal_embedding import classification_model
from tensorflow.io import gfile




PyTree = Any

# Match PyTorch default LayerNorm epsilon of 1e-5 (FLAX defaults to 1e-6).
LayerNorm = functools.partial(nn.LayerNorm, epsilon=1e-5)



def quick_gelu(x: jnp.ndarray) -> jnp.ndarray:
  return x * jax.nn.sigmoid(1.702 * x)



class AttentionPool(nn.Module):
  """Attention pooling layer.

  Attributes:
    num_heads: Number of heads.
    features: Number of features.
  """
  num_heads: int
  features: Optional[int] = None

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    x = x.reshape(x.shape[0], -1, x.shape[3])

    x = jnp.concatenate([x.mean(axis=1, keepdims=True), x], axis=1)

    positional_embedding = self.param(
        'positional_embedding',
        jax.nn.initializers.normal(1. / x.shape[-1]**0.5),
        (x.shape[1], x.shape[2]))
    attn = nn.MultiHeadDotProductAttention(
        self.num_heads,
        qkv_features=x.shape[-1],
        use_bias=True,
        out_features=self.features,
        name='attn')

    x = x + positional_embedding[jnp.newaxis].astype(x.dtype)
    x = attn(x[:, :1], x)
    return x[:, 0]



class MLP(nn.Module):
  """Simple MLP for Transformer."""

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    ch = x.shape[-1]
    x = nn.Dense(4 * ch, name='c_fc')(x)
    x = quick_gelu(x)
    x = nn.Dense(ch, name='c_proj')(x)
    return x



class ResidualAttentionBlock(nn.Module):
  """Self-attention block of Transformer.

  Attributes:
    num_heads: Number of heads.
  """
  num_heads: int

  @nn.compact
  def __call__(
      self, x: jnp.ndarray, train: bool, attn_mask=None
  ) -> jnp.ndarray:
    xn = LayerNorm(name='ln_1')(x)
    x = x + nn.SelfAttention(self.num_heads, name='attn', deterministic=train)(
        xn, attn_mask
    )
    xn = LayerNorm(name='ln_2')(x)
    x = x + MLP(name='mlp')(xn)
    return x



class Transformer(nn.Module):
  """Transformer module.

  Attributes:
    features: Number of features.
    num_layers: Number of layers for each block.
    num_heads: Number of heads.
    use_underscore_module_name: Optionally replace '.' with '_' in parameter
      naming for PAX checkpoint loading.
  """
  features: int
  num_layers: int
  num_heads: int
  use_underscore_module_name: bool = False

  @nn.compact
  def __call__(self,
               x: jnp.ndarray,
               train: bool,
               attn_mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    def _n(name):
      """A helper function that optionally replace '.' with '_'."""
      if self.use_underscore_module_name:
        return name.replace('.', '_')
      else:
        return name

    for i in range(self.num_layers):
      x = ResidualAttentionBlock(
          num_heads=self.num_heads, name=_n(f'resblocks.{i}')
      )(x, train, attn_mask)
    return x



class ClipVisionTransformer(nn.Module):
  """Vision Transformer.

  Attributes:
    patch_size: The size of the patches to embed.
    features: Number of features.
    num_layers: Number of transformer blocks (self-attn + MLP).
    num_heads: Number of attention heads.
    out_features: Number of output features. If None, return transformer output.
    use_underscore_module_name: Optionally replace '.' with '_' in parameter
      naming for PAX checkpoint loading.
  """
  patch_size: int
  features: int
  num_layers: int
  num_heads: int
  out_features: Optional[int]
  num_classes: int
  use_underscore_module_name: bool = False
  output_dim: int = -1

  @nn.compact
  def __call__(
      self,
      x: jnp.ndarray,
      train: bool,
      init: bool = False,
      debug: bool = False,
      return_feats: bool = False,
      project_feats: bool = True,
  ) -> jnp.ndarray:
  
    outputs = {}
    outputs['embeddings'] = {}

    if train or init:
      outputs['classifier'] = {}

    x = nn.Conv(self.features,
                kernel_size=(self.patch_size, self.patch_size),
                strides=(self.patch_size, self.patch_size),
                use_bias=False, name='conv1')(x)
    x = x.reshape(x.shape[0], -1, x.shape[-1])
    scale = 1.0 / jnp.sqrt(self.features)
    class_embedding = self.param('class_embedding',
                                 jax.nn.initializers.normal(stddev=scale),
                                 (self.features,))
    x = jnp.concatenate((jnp.tile(class_embedding[None, None, :],
                                  (x.shape[0], 1, 1)), x),
                        axis=1)
    positional_embedding = self.param('positional_embedding',
                                      jax.nn.initializers.normal(stddev=scale),
                                      (x.shape[1], self.features))
    x = x + positional_embedding[None]

    x = LayerNorm(name='ln_pre')(x)
    x = Transformer(
        features=self.features,
        num_layers=self.num_layers,
        num_heads=self.num_heads,
        use_underscore_module_name=self.use_underscore_module_name,
        name='transformer')(
            x, train)

    x = LayerNorm(name='ln_post')(x)

    # Normalize the output
    x = x[:, 0]
    x /= jnp.linalg.norm(x, ord=2, axis=1, keepdims=True)

    outputs['embeddings']['backbone_out'] = x

    if project_feats:

      # our projection layer for dim reduction.
      if self.output_dim > 0:
        x = nn.Dense(self.output_dim, name='projection')(x)
      else:
        x = nn_layers.IdentityLayer(name='projection')(x)

      # l2 norm the embeddings (again)
      x /= jnp.linalg.norm(x, ord=2, axis=1, keepdims=True)

      outputs['embeddings']['projected'] = x #TODO: use of project_feats flag is not needed anymore since they are two different named embeddings now


    if not return_feats:  # pass through classification layer
      x = nn.Dense(
          self.num_classes,
          use_bias=False,
          kernel_init=nn.initializers.lecun_uniform(),
          name='output_projection',
      )(x)
      weights_norms = jnp.linalg.norm(
          self.variables['params']['output_projection']['kernel'], axis=0
      )  # norms of class prototypes
      x /= weights_norms

      outputs['classifier']['logits'] = x

    return outputs




def _convert_attn_layers(params: Mapping[str, np.ndarray],
                         dim_head: int = 64) -> PyTree:
  """Convert attention parameters."""
  new_params = {}
  processed_attn_layers = []
  for k, v in params.items():
    if 'attn.' in k:
      base = k[:k.rindex('attn.')+5]
      if base in processed_attn_layers:
        continue
      processed_attn_layers.append(base)
      dim = params[base + 'out_proj.bias'].shape[-1]
      heads = dim // dim_head
      new_params[base + 'out.weight'] = params[
          base + 'out_proj.weight'].T.reshape(heads, dim_head, dim)
      new_params[base + 'out.bias'] = params[base + 'out_proj.bias']
      qkv_bias = params[base + 'in_proj_bias'].reshape(3, heads, dim_head)
      qkv_kernel = np.transpose(params[base + 'in_proj_weight'].reshape(
          3, heads, dim_head, dim), (0, 3, 1, 2))
      for i, kk in enumerate(('query', 'key', 'value')):
        new_params[base + f'{kk}.bias'] = qkv_bias[i]
        new_params[base + f'{kk}.weight'] = qkv_kernel[i]
    else:
      new_params[k] = v
  return new_params



class ViTWithEmbeddingClassificationModel(
    classification_model.UniversalEmbeddingClassificationModel
):
  """Vision Transformer model for classification task."""

  # overriding the method of scenic
  def build_flax_model(self) -> nn.Module:
    return ClipVisionTransformer(
        patch_size=self.config.model.patches.size[0],
        features=self.config.model.hidden_size,
        num_layers=self.config.model.num_layers,
        num_heads=self.config.model.num_heads,
        num_classes=self.dataset_meta_data['num_classes'],
        output_dim=self.config.model.output_dim,
        use_underscore_module_name=False,
        out_features=None
    )

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    return ml_collections.ConfigDict(
        {
            'model': dict(
                patch_size=4,
                features=16,
                num_layers=1,
                num_heads=2
            )
        }
    )

  def load_model_vars(
      self, train_state: Any, checkpoint_path: str, dim_head: int = 64
  ) -> Any:
    """Convert torch parameters to flax parameters."""
    # Expand QKV dense input projection to separate Q, K, V projections
    # and fix shape/transposing of attention layers.
    with gfile.GFile(checkpoint_path, 'rb') as f:
      torch_vars = np.load(f, allow_pickle=True).tolist()

    for var_key in torch_vars:
      torch_vars[var_key] = torch_vars[var_key].astype(np.float32)

    torch_vars = _convert_attn_layers(torch_vars, dim_head)
    flax_vars = {}
    torch_vars.pop('context_length', None)
    torch_vars.pop('input_resolution', None)
    torch_vars.pop('vocab_size', None)
    for torch_key, v in torch_vars.items():
      if 'num_batches_tracked' in torch_key:
        continue

      if 'conv' in torch_key or 'downsample.0.weight' in torch_key:
        v = v.transpose(2, 3, 1, 0)
      elif (
          'weight' in torch_key and v.ndim == 2 and 'embedding' not in torch_key
      ):
        # Fully connected layers are transposed, embeddings are not
        v = v.T

      jax_key = torch_key.replace('visual.proj', 'visual.proj.kernel')
      jax_key = jax_key.replace('text_projection', 'text_projection.kernel')
      if 'bn' in jax_key or 'ln' in jax_key or 'downsample.1' in jax_key:
        jax_key = jax_key.replace('.weight', '.scale')
      else:
        jax_key = jax_key.replace('.weight', '.kernel')
      if (jax_key.startswith('transformer') or
          jax_key.startswith('text_projection') or
          jax_key.startswith('ln_final') or
          jax_key.startswith('positional_embedding')):
        jax_key = 'text.' + jax_key

      jax_key = jax_key.replace(
          'token_embedding.kernel', 'text.token_embedding.embedding')

      jax_key = jax_key.replace('attnpool.k_proj', 'attnpool.attn.key')
      jax_key = jax_key.replace('attnpool.q_proj', 'attnpool.attn.query')
      jax_key = jax_key.replace('attnpool.v_proj', 'attnpool.attn.value')
      jax_key = jax_key.replace('attnpool.c_proj', 'attnpool.attn.out')
      if 'attnpool.attn.out' in jax_key:
        if jax_key.endswith('kernel'):
          v = v.reshape(-1, dim_head, v.shape[-1])
      elif 'attnpool.attn' in jax_key:
        if jax_key.endswith('bias'):
          v = v.reshape(-1, dim_head)
        else:
          v = v.reshape(v.shape[0], -1, dim_head)

      if jax_key.endswith('running_mean'):
        jax_key = 'batch_stats.' + jax_key.replace('.running_mean', '.mean')
      elif jax_key.endswith('running_var'):
        jax_key = 'batch_stats.' + jax_key.replace('.running_var', '.var')
      else:
        jax_key = 'params.' + jax_key

      jax_key = jax_key.replace('.', '/')
      jax_key = jax_key.replace('resblocks/', 'resblocks.')
      jax_key = jax_key.replace('resblocks/', 'resblocks.')
      jax_key = jax_key.split('/')[1:]
      if jax_key[0] == 'text':
        continue
      elif jax_key[0] == 'visual':
        jax_key.pop(0)

      flax_vars[tuple(jax_key)] = jnp.asarray(v)
    flax_vars = flax.traverse_util.unflatten_dict(flax_vars)

    params = flax.core.unfreeze(train_state.params)
    for key, param in flax_vars.items():
      if key not in params:
        continue
      params[key] = param

    return train_state.replace(params=flax.core.freeze(params))