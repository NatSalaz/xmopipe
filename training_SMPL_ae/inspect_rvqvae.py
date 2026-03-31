import torch
import yaml
import argparse

from models.rvqvae.rvqvae import RVQVAE


def inspect_model(config, checkpoint_path):
    """
    Inspect the RVQVAE model structure to understand its attributes and methods
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("RVQVAE MODEL INSPECTION")
    print("=" * 80)

    # Load model
    print("\n1. Creating model from config...")
    net = RVQVAE(config)

    print("\n2. Loading checkpoint...")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    net.load_state_dict(ckpt["model_state_dict"], strict=False)
    net.to(device).eval()

    print("\n3. Model architecture:")
    print(net)

    print("\n" + "=" * 80)
    print("4. Main model attributes:")
    print("=" * 80)
    for attr in dir(net):
        if not attr.startswith("_") and not callable(getattr(net, attr)):
            print(f"  - {attr}: {type(getattr(net, attr))}")

    print("\n" + "=" * 80)
    print("5. Main model methods:")
    print("=" * 80)
    for attr in dir(net):
        if not attr.startswith("_") and callable(getattr(net, attr)):
            print(f"  - {attr}()")

    print("\n" + "=" * 80)
    print("6. Quantizer details:")
    print("=" * 80)
    if hasattr(net, "quantizer"):
        quantizer = net.quantizer
        print(f"Quantizer type: {type(quantizer)}")
        print(f"Quantizer class: {quantizer.__class__.__name__}")

        print("\n  Quantizer attributes:")
        for attr in dir(quantizer):
            if not attr.startswith("_") and not callable(getattr(quantizer, attr)):
                try:
                    val = getattr(quantizer, attr)
                    if isinstance(val, (int, float, str, bool)):
                        print(f"    - {attr}: {val}")
                    else:
                        print(f"    - {attr}: {type(val)}")
                except:
                    print(f"    - {attr}: <unable to access>")

        print("\n  Quantizer methods:")
        for attr in dir(quantizer):
            if not attr.startswith("_") and callable(getattr(quantizer, attr)):
                print(f"    - {attr}()")

        # Check for residual quantizers (RVQ = Residual Vector Quantization)
        if hasattr(quantizer, "quantizers"):
            print(f"\n  ✓ Found 'quantizers' attribute (likely residual VQ)")
            quantizers = quantizer.quantizers
            print(f"    Number of quantizers: {len(quantizers)}")
            if len(quantizers) > 0:
                print(f"\n    First quantizer details:")
                q0 = quantizers[0]
                print(f"      Type: {type(q0)}")
                for attr in [
                    "num_embeddings",
                    "embedding_dim",
                    "codebook_size",
                    "embedding",
                ]:
                    if hasattr(q0, attr):
                        val = getattr(q0, attr)
                        if isinstance(val, (int, float, str, bool)):
                            print(f"      - {attr}: {val}")
                        else:
                            print(
                                f"      - {attr}: {type(val)} shape={getattr(val, 'shape', 'N/A')}"
                            )

    print("\n" + "=" * 80)
    print("7. Encoder/Decoder:")
    print("=" * 80)
    if hasattr(net, "encoder"):
        print(f"  ✓ Has encoder: {type(net.encoder)}")
    if hasattr(net, "decoder"):
        print(f"  ✓ Has decoder: {type(net.decoder)}")

    print("\n" + "=" * 80)
    print("8. Testing forward pass:")
    print("=" * 80)

    # Test with dummy input
    batch_size = 2
    seq_len = 64
    input_dim = config.get("input_dim", 263)

    dummy_input = torch.randn(batch_size, seq_len, input_dim).to(device)
    print(f"  Input shape: {dummy_input.shape}")

    try:
        with torch.no_grad():
            output = net(dummy_input)

        if isinstance(output, tuple):
            print(f"  Output is tuple with {len(output)} elements:")
            for i, o in enumerate(output):
                if isinstance(o, torch.Tensor):
                    print(f"    [{i}] Tensor: shape={o.shape}, dtype={o.dtype}")
                elif isinstance(o, dict):
                    print(f"    [{i}] Dict with keys: {list(o.keys())}")
                else:
                    print(f"    [{i}] {type(o)}")
        else:
            print(f"  Output shape: {output.shape}")
    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")

    print("\n" + "=" * 80)
    print("9. Testing encode:")
    print("=" * 80)

    try:
        with torch.no_grad():
            encoded = net.encode(dummy_input)

        if isinstance(encoded, tuple):
            print(f"  Encode output is tuple with {len(encoded)} elements:")
            for i, e in enumerate(encoded):
                if isinstance(e, torch.Tensor):
                    print(f"    [{i}] Tensor: shape={e.shape}, dtype={e.dtype}")
                elif isinstance(e, dict):
                    print(f"    [{i}] Dict with keys: {list(e.keys())}")
                elif isinstance(e, list):
                    print(f"    [{i}] List with {len(e)} elements")
                else:
                    print(f"    [{i}] {type(e)}")
        else:
            print(f"  Encode output shape: {encoded.shape}")
    except Exception as e:
        print(f"  ✗ Encode failed: {e}")

    print("\n" + "=" * 80)
    print("10. Config used:")
    print("=" * 80)
    for key, val in config.items():
        print(f"  {key}: {val}")

    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect RVQVAE model structure")
    parser.add_argument("--config", type=str, required=True, help="Config YAML file")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Model checkpoint"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    inspect_model(config, args.checkpoint)
