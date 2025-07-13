from src.models.experiment_models.model_PTM import Cd_PTM_Model
from src.models.experiment_models.model_DTM import Cd_DTM_Model


def get_model(
    model_type: str = "ptm",
    slice_input_dim: int = 2,
    slice_emb_dim: int = 256,
    transformer_hidden_dim: int = 256,
    transformer_layers: int = 2,
    transformer_heads: int = 1,
    transformer_dropout: float = 0.1,
    max_num_slices: int = 80,
    k_neighbors: int = 20,  # Only used by DGCNN
):
    """
    Return an instantiated Cd model based on model_type.

    Supported:
        - "plm" → PointNet + LSTM + MLP
        - "ptm" → PointNet + Transformer + MLP
        - "dtm" → DGCNN + Transformer + MLP (uses dynamic slice batching)

    Returns:
        nn.Module
    """
    if model_type == "plm":
        return Cd_PTM_Model(
            slice_input_dim=slice_input_dim,
            slice_emb_dim=slice_emb_dim,
            transformer_hidden_dim=transformer_hidden_dim,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_dropout=transformer_dropout,
            max_num_slices=max_num_slices,
        )
    elif model_type == "ptm":
        return Cd_PTM_Model(
            slice_input_dim=slice_input_dim,
            slice_emb_dim=slice_emb_dim,
            transformer_hidden_dim=transformer_hidden_dim,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_dropout=transformer_dropout,
            max_num_slices=max_num_slices,
        )
    elif model_type == "dtm":
        return Cd_DTM_Model(
            slice_input_dim=slice_input_dim,
            slice_emb_dim=slice_emb_dim,
            transformer_hidden_dim=transformer_hidden_dim,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_dropout=transformer_dropout,
            max_num_slices=max_num_slices,
            k_neighbors=k_neighbors,
        )
    else:
        raise ValueError(f"Unsupported model_type '{model_type}'. Use 'ptm' or 'dtm'.")
