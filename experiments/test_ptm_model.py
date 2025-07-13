import torch
from src.models.experiment_models.model_PTM import Cd_PTM_Model


def test_ptm_forward_pass():
    # Instantiate the model
    model = Cd_PTM_Model(
        slice_input_dim=2,
        slice_emb_dim=256,
        transformer_hidden_dim=256,
        transformer_layers=2,
        transformer_heads=2,
        transformer_dropout=0.1,
        max_num_slices=80,
    )

    # Get dummy input
    x = model.example_input(batch_size=4, S=80, P=6500)

    # Move to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = model.to(device)
    x = tuple(t.to(device) for t in x)

    # Run forward pass
    with torch.no_grad():
        output = model(x)

    print("✅ Forward pass successful")
    print("Output shape:", output.shape)  # Expect (B,)
    print("Output values:", output)


if __name__ == "__main__":
    test_ptm_forward_pass()
