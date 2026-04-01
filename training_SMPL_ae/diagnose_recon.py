import torch
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt

from models.rvqvae.rvqvae import RVQVAE
from data.motion_loader import DATALoader, MotionDataset
from utils.motion_process import recover_from_ric
from utils.metric_utils import calc_mpjpe, calc_pampjpe


def diagnose_reconstruction(config, checkpoint_path):
    """
    Detailed reconstruction diagnostic to identify issues.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    print("=" * 80)
    print("RVQVAE RECONSTRUCTION DIAGNOSTIC")
    print("=" * 80)

    # Load model
    print("\n1. Loading model...")
    net = RVQVAE(config)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    net.load_state_dict(ckpt["model_state_dict"], strict=False)
    net.to(device).eval()
    print(f"   - Model loaded (epoch {ckpt['epoch']}, step {ckpt['global_step']})")

    # Load data
    print("\n2. Loading data...")
    loader = DATALoader(
        dataset_name=config["dataset_name"],
        batch_size=4,  # Small batch for diagnostic
        num_workers=0,
        window_size=config.get("window_size", 64),
        data_root=config.get("data_root"),
        shuffle=False,
        data_split="test",
        normalized=False,
    )

    data_trained = MotionDataset(dataset_name=config["dataset_name"], subsampling=0.0)

    mean = torch.from_numpy(data_trained.mean).to(device)
    std = torch.from_numpy(data_trained.std).to(device)

    print(f"   - Mean shape: {mean.shape}, std shape: {std.shape}")
    print(f"   - Mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"   - Std range: [{std.min():.4f}, {std.max():.4f}]")

    joints_num = 22

    # Get one batch
    print("\n3. Testing on a batch...")
    motions = next(iter(loader)).to(device)
    B, T, D = motions.shape
    print(f"   - Batch shape: {motions.shape}")
    print(f"   - Input range: [{motions.min():.4f}, {motions.max():.4f}]")

    # Normalize
    motions_normalized = (motions - mean) / std
    print(
        f"   - Normalized range: [{motions_normalized.min():.4f}, {motions_normalized.max():.4f}]"
    )

    # Forward pass
    print("\n4. Forward pass...")
    with torch.no_grad():
        output = net(motions_normalized)
        recon_normalized = output[0]

    print(f"   - Reconstruction (normalized) shape: {recon_normalized.shape}")
    print(
        f"   - Reconstruction (normalized) range: [{recon_normalized.min():.4f}, {recon_normalized.max():.4f}]"
    )

    # Denormalize
    recon = recon_normalized * std + mean
    print(
        f"   - Reconstruction (denormalized) range: [{recon.min():.4f}, {recon.max():.4f}]"
    )

    # Test encode/decode separately
    print("\n5. Testing encode/decode separately...")
    with torch.no_grad():
        # Encode
        indices, z_q = net.encode(motions_normalized)
        print(f"   - Indices shape: {indices.shape}")
        print(f"   - Indices range: [{indices.min()}, {indices.max()}]")
        print(f"   - Latent (z_q) shape: {z_q.shape}")

        # Decode via forward_decoder (using indices, not z_q!)
        if hasattr(net, "forward_decoder"):
            recon2_normalized = net.forward_decoder(indices)
            print(
                f"   - Decode via forward_decoder (from indices): {recon2_normalized.shape}"
            )
        else:
            # Manual decode using z_q embeddings
            z_q_sum = z_q.sum(dim=0)  # [B, latent_dim, T']
            recon2_normalized = net.decoder(z_q_sum)  # [B, latent_dim, T]
            recon2_normalized = recon2_normalized.transpose(1, 2)  # [B, T, latent_dim]
            print(f"   - Decode manual (from z_q): {recon2_normalized.shape}")

        recon2 = recon2_normalized * std + mean
        print(f"   - Encode->Decode range: [{recon2.min():.4f}, {recon2.max():.4f}]")

    # Compare forward vs encode/decode
    print("\n6. Comparing forward() vs encode()+decode()...")
    diff = (recon - recon2).abs().mean()
    print(f"   - Mean difference: {diff:.6f}")
    if diff < 1e-5:
        print("   - Both methods give the same result!")
    else:
        print("   ⚠ WARNING: The two methods differ!")

    # Recovery to 3D
    print("\n7. Converting to 3D joints...")
    print(
        f"   Input motion sample (first timestep, first 10 dims): {motions[0, 0, :10]}"
    )
    print(f"   Recon motion sample (first timestep, first 10 dims): {recon[0, 0, :10]}")

    gt_xyz = recover_from_ric(motions, joints_num)
    pred_xyz = recover_from_ric(recon, joints_num)

    print(f"   - GT 3D shape: {gt_xyz.shape}")
    print(f"   - Pred 3D shape: {pred_xyz.shape}")
    print(f"   - GT 3D range: [{gt_xyz.min():.4f}, {gt_xyz.max():.4f}]")
    print(f"   - Pred 3D range: [{pred_xyz.min():.4f}, {pred_xyz.max():.4f}]")

    # Compute metrics on first sample
    print("\n8. Metrics on first sample...")
    gt_sample = gt_xyz[0]  # [T, J, 3]
    pred_sample = pred_xyz[0]  # [T, J, 3]

    mpjpe = calc_mpjpe(gt_sample, pred_sample)
    pampjpe = calc_pampjpe(pred_sample, gt_sample)

    print(f"   - MPJPE: {mpjpe.mean()*1000:.4f} mm")
    print(f"   - PA-MPJPE: {pampjpe.mean()*1000:.4f} mm")

    # Position analysis
    print("\n9. 3D position analysis...")
    gt_center = gt_sample.mean(dim=1)  # [T, 3] - center of mass
    pred_center = pred_sample.mean(dim=1)

    print(f"   GT center range: [{gt_center.min():.4f}, {gt_center.max():.4f}]")
    print(f"   Pred center range: [{pred_center.min():.4f}, {pred_center.max():.4f}]")

    center_diff = (gt_center - pred_center).abs().mean(dim=0)
    print(f"   Center difference (X, Y, Z): {center_diff.cpu().numpy()*1000} mm")

    # Joint spread analysis
    gt_spread = (gt_sample - gt_sample.mean(dim=1, keepdim=True)).abs().mean()
    pred_spread = (pred_sample - pred_sample.mean(dim=1, keepdim=True)).abs().mean()

    print(f"   GT joint spread: {gt_spread*1000:.4f} mm")
    print(f"   Pred joint spread: {pred_spread*1000:.4f} mm")

    # Check if reconstruction is too smooth
    print("\n10. Temporal variance analysis...")
    gt_temporal_var = motions[0].var(dim=0).mean()
    recon_temporal_var = recon[0].var(dim=0).mean()

    print(f"   GT temporal variance: {gt_temporal_var:.6f}")
    print(f"   Recon temporal variance: {recon_temporal_var:.6f}")
    print(f"   Ratio: {recon_temporal_var/gt_temporal_var:.4f}")

    # Feature-wise comparison
    print("\n11. Dimension-by-dimension comparison...")
    feature_mse = ((motions[0] - recon[0]) ** 2).mean(dim=0)
    worst_features = torch.argsort(feature_mse, descending=True)[:10]

    print("   Top 10 worst dimensions (MSE):")
    for i, idx in enumerate(worst_features):
        print(f"      [{i+1}] Dim {idx}: MSE={feature_mse[idx]:.6f}")

    # Check training loss from checkpoint
    print("\n12. Checkpoint info...")
    if "train_loss" in ckpt:
        print(f"   Training loss: {ckpt['train_loss']:.6f}")
    if "val_loss" in ckpt:
        print(f"   Validation loss: {ckpt['val_loss']:.6f}")

    # Visualize one joint trajectory
    print("\n13. Saving trajectories for visualization...")

    # Pick one joint (e.g., right hand = joint 21)
    joint_idx = 21

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    fig.suptitle(f"Joint {joint_idx} Trajectory Comparison")

    for i, (ax, coord) in enumerate(zip(axes, ["X", "Y", "Z"])):
        gt_traj = gt_sample[:, joint_idx, i].cpu().numpy()
        pred_traj = pred_sample[:, joint_idx, i].cpu().numpy()

        ax.plot(gt_traj, label="Ground Truth", linewidth=2)
        ax.plot(pred_traj, label="Reconstruction", linewidth=2, linestyle="--")
        ax.set_ylabel(f"{coord} (meters)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Frame")
    plt.tight_layout()
    plt.savefig("./trajectory_comparison.png", dpi=150)
    print("   - Saved to trajectory_comparison.png")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)

    # Summary
    print("\nSUMMARY:")
    print(f"   MPJPE: {mpjpe.mean()*1000:.2f} mm")
    print(f"   PA-MPJPE: {pampjpe.mean()*1000:.2f} mm")
    print(f"   Variance ratio: {recon_temporal_var/gt_temporal_var:.2f}")

    if mpjpe.mean() * 1000 > 50:
        print("\nPOTENTIAL ISSUES:")
        if (recon_temporal_var / gt_temporal_var) < 0.5:
            print("   • Reconstruction is too smooth (variance < 50% of GT)")
        if center_diff.mean() * 1000 > 20:
            print("   • Center of mass is poorly reconstructed")
        if abs(gt_spread - pred_spread) > 0.1:
            print("   • Joint scale is incorrect")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RVQVAE reconstruction diagnostic")
    parser.add_argument("--config", type=str, required=True, help="Config YAML file")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Model checkpoint"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    diagnose_reconstruction(config, args.checkpoint)
