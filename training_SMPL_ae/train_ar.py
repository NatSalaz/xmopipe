import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import yaml
from pathlib import Path

from data.motion_loader import DATALoader
from training.trainer import Trainer
from models.vae import VAE
from models.vqvae.vqvae import VQVAE
from models.rvqvae.rvqvae import RVQVAE
from models.klvae.autoencoder import AutoencoderKL
from training.loss_manager import (
    LossManager,
    create_vae_loss_manager,
    create_vqvae_loss_manager,
    create_rvqvae_loss_manager,
    create_klvae_loss_manager,
)


def get_model(model_name, config):
    """Factory to create model"""
    input_dim = config["input_dim"]
    latent_dim = config["latent_dim"]

    if model_name == "vae":
        return VAE(input_dim=input_dim, latent_dim=latent_dim)
    elif model_name == "vqvae":
        return VQVAE(config=config)
    elif model_name == "rvqvae":
        return RVQVAE(config=config)
    elif model_name == "klvae":
        return AutoencoderKL()
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_loss_manager(model_name, config):
    """Factory to create LossManager for each model type"""
    if model_name == "vqvae":
        return create_vqvae_loss_manager(
            recon_weight=config.get("recon_weight", 1.0),
            vq_weight=config.get("vq_weight", 0.02),
            commitment=config.get("commitment_weight", 0.25),
        )
    if model_name == "rvqvae":
        return create_rvqvae_loss_manager(
            recon_weight=config.get("recon_weight", 1.0),
            vq_weight=config.get("vq_weight", 0.02),
            commitment=config.get("commitment_weight", 0.25),
        )
    elif model_name == "klvae":
        return create_klvae_loss_manager(beta=config.get("beta", 1.0))
    elif model_name == "vae":
        return create_vae_loss_manager(beta=config.get("beta", 1.0))
    else:
        raise ValueError(f"Unknown model: {model_name}")


def load_config(config_path):
    """Load config from YAML"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    config = load_config(args.config)

    # Set random seed for reproducibility
    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # DataLoaders
    print("\nLoading data...")
    train_loader, val_loader = DATALoader(
        dataset_name=config["dataset_name"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        window_size=config["window_size"],
        data_root=config.get("data_root", None),
        shuffle=True,
        subsampling=config.get("subsampling", None),
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    sample_batch = next(iter(train_loader))
    input_dim = sample_batch.shape[-1]  # (batch, seq_len, features)
    seq_len = sample_batch.shape[1]
    config["input_dim"] = input_dim
    config["seq_len"] = seq_len

    print(f"Input shape: {sample_batch.shape}")
    print(f"Features: {input_dim}, Sequence length: {seq_len}")
    print(f"\nCreating model: {config['model_type']}")
    model = get_model(config["model_type"], config)
    print(model)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 1e-4),
    )

    # Learning rate scheduler (optional)

    scheduler = None
    if config.get("use_scheduler", False):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )

    # Loss Manager
    print(f"\nConfiguring losses for {config['model_type']}...")
    loss_manager = get_loss_manager(config["model_type"], config)
    print(f"Loss configuration: {loss_manager}")
    print(f"Active losses: {list(loss_manager.get_summary().keys())}")

    # Experiment name
    experiment_name = (
        f"{config['model_type']}_{config['dataset_name']}_{args.exp_suffix}"
        if args.exp_suffix
        else f"{config['model_type']}_{config['dataset_name']}"
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_manager=loss_manager,
        device=device,
        experiment_name=experiment_name,
        config=config,
        log_dir=args.log_dir,
        max_steps=300000,
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train(
        num_epochs=config["num_epochs"],
        early_stopping_patience=config.get("early_stopping_patience", None),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train motion reconstruction models")

    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )

    parser.add_argument(
        "--exp-suffix", type=str, default="", help="Suffix for experiment name"
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="./experiments",
        help="Directory to save experiments",
    )

    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )

    args = parser.parse_args()
    main(args)
