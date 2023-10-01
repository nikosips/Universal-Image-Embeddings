from universal_embedding import vit_with_embedding
from universal_embedding import clip_vit_with_embedding


MODELS = {
    'vit_with_embedding': vit_with_embedding.ViTWithEmbeddingClassificationModel,
    'clip_vit_with_embedding': clip_vit_with_embedding.ViTWithEmbeddingClassificationModel,
}