import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json
import os
from pathlib import Path
import time
from datetime import datetime
import numpy as np
from training.loss_manager import LossManager


class Trainer:
    """
    Includes: TensorBoard, logging JSON, checkpointing, early stopping.
    """

    def __init__(
        self,
        model,
        val_loader,
        train_loader,
        optimizer=None,
        loss_manager=None,
        device="cuda",
        experiment_name=None,
        config=None,
        log_dir="./experiments",
        max_steps=3000000,
    ):
        self.val_loader = val_loader
        self.train_loader = train_loader
        self.model = model.to(device)
        self.optimizer = optimizer
        self.config = config or {}
        self.max_steps = max_steps
        if loss_manager is None:
            print("No LossManager given, creating default MSE loss.")

            loss_manager = LossManager(self.config)
            loss_manager.add_loss("mse", nn.MSELoss(), weight=1.0)
        self.loss_manager = loss_manager
        self.device = device

        # Setup experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = experiment_name or f"exp_{timestamp}"
        self.exp_dir = Path(log_dir) / exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.log_dir = self.exp_dir / "logs"
        self.tb_dir = self.exp_dir / "tensorboard"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        self.tb_dir.mkdir(exist_ok=True)

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.tb_dir))

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.save_every_steps = 10000
        self.val_every_steps = 2000
        self.early_stopping_patience_steps = 5
        self.steps_without_improvement = 0

        # History for logging
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rates": [],
            "epoch_times": [],
        }

        # Save config
        self._save_config()

    def _save_config(self):
        config_path = self.exp_dir / "config.json"
        config_dict = {
            "model_name": self.model.__class__.__name__,
            "model_params": sum(p.numel() for p in self.model.parameters()),
            "optimizer": self.optimizer.__class__.__name__ if self.optimizer else None,
            "loss_manager_summary": self.loss_manager.get_summary(),
            "device": str(self.device),
            **self.config,
        }
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)

    def _save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": (
                self.optimizer.state_dict() if self.optimizer else None
            ),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }

        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "latest.pth"
        torch.save(checkpoint, latest_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            torch.save(checkpoint, best_path)
            print(f"Best model saved! Val Loss: {self.best_val_loss:.6f}")
            return
        # Save epoch checkpoint every N epochs
        # if epoch % self.config.get("save_every", 100) == 0:
        #    epoch_path = self.checkpoint_dir / f"epoch_{epoch:04d}.pth"
        #    torch.save(checkpoint, epoch_path)
        if self.save_every_steps:
            step_path = self.checkpoint_dir / f"step_{self.global_step:07d}.pth"
            torch.save(checkpoint, step_path)

    def load_checkpoint(self, checkpoint_path):
        """Charge un checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if self.optimizer and checkpoint["optimizer_state_dict"]:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.history = checkpoint["history"]
        print(f"Checkpoint loaded from epoch {self.current_epoch}")

    def train_epoch(self):
        """Train the model for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)

        accumulated_losses = {}

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, data in enumerate(pbar):
            if self.global_step >= self.max_steps:
                print(f"\nMax steps ({self.max_steps}) reached. Stopping training.")
                break

            data = data.to(self.device)
            self.optimizer.zero_grad()

            if (
                self.config.get("model_type", "vae").lower() == "vqvae"
                or self.config.get("model_type", "vae").lower() == "rvqvae"
            ):
                x_recon, vq_output, perplexity = self.model(data)
                loss, loss_details = self.loss_manager.compute(
                    predictions=x_recon,
                    targets=data,
                    perplexity=perplexity,
                    quantized=vq_output,
                    inputs=data,
                )
            elif self.config.get("model_type", "vae").lower() == "klvae":
                x_recon, posterior = self.model(data)
                loss, loss_details = self.loss_manager.compute(
                    predictions=x_recon,
                    targets=data,
                    posterior=posterior,
                )
            else:
                x_recon, mu, logvar = self.model(data)
                loss, loss_details = self.loss_manager.compute(
                    predictions=x_recon, targets=data, mu=mu, logvar=logvar
                )

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.get("grad_clip", None):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["grad_clip"]
                )

            self.optimizer.step()

            total_loss += loss_details["total"]
            for loss_name, loss_value in loss_details.items():
                if loss_name not in accumulated_losses:
                    accumulated_losses[loss_name] = 0
                accumulated_losses[loss_name] += loss_value

            avg_loss = total_loss / (batch_idx + 1)

            postfix = {"loss": f"{loss_details['total']:.6f}", "avg": f"{avg_loss:.6f}"}
            for name, value in loss_details.items():
                if not name.endswith("_weighted") and name != "total":
                    postfix[name] = f"{value:.4f}"
            pbar.set_postfix(postfix)
            if self.global_step % self.config.get("log_every", 10) == 0:
                for loss_name, loss_value in loss_details.items():
                    self.writer.add_scalar(
                        f"Train/{loss_name}", loss_value, self.global_step
                    )

            self.global_step += 1
            if self.save_every_steps and self.global_step % self.save_every_steps == 0:
                print(f"\n[Checkpoint] Saving at step {self.global_step}")
                self._save_checkpoint(epoch=self.current_epoch)
            if self.val_every_steps and self.global_step % self.val_every_steps == 0:
                should_stop = self.validate_and_check_early_stop(
                    epoch=self.current_epoch
                )
                if should_stop:
                    print("Early stopping with steps")
                    return None
        epoch_losses = {
            name: value / num_batches for name, value in accumulated_losses.items()
        }
        print("Step:", self.global_step)
        return epoch_losses

    @torch.no_grad()
    def validate(self):
        if self.val_loader is None:
            return None

        self.model.eval()
        num_batches = len(self.val_loader)

        accumulated_losses = {}

        for data in tqdm(self.val_loader, desc="Validation"):
            data = data.to(self.device)

            if (
                self.config.get("model_type", "vae").lower() == "vqvae"
                or self.config.get("model_type", "vae").lower() == "rvqvae"
            ):
                x_recon, vq_output, perplexity = self.model(data)
                _, loss_details = self.loss_manager.compute(
                    predictions=x_recon,
                    targets=data,
                    perplexity=perplexity,
                    quantized=vq_output,
                    inputs=data,
                )
            elif self.config.get("model_type", "vae").lower() == "klvae":
                x_recon, posterior = self.model(data)
                _, loss_details = self.loss_manager.compute(
                    predictions=x_recon,
                    targets=data,
                    posterior=posterior,
                )
            else:
                recon, mu, logvar = self.model(data)
                _, loss_details = self.loss_manager.compute(
                    predictions=recon, targets=data, mu=mu, logvar=logvar
                )

            # Accumulation
            for loss_name, loss_value in loss_details.items():
                if loss_name not in accumulated_losses:
                    accumulated_losses[loss_name] = 0
                accumulated_losses[loss_name] += loss_value

        metrics = {
            name: value / num_batches for name, value in accumulated_losses.items()
        }

        return metrics

    def train(self, num_epochs, early_stopping_patience=None):
        """
        Args:
            num_epochs: number of epochs
            early_stopping_patience: Patience for early stopping (None = no early stopping)
        """
        print(f"\n{'='*60}")
        print(f"Starting Training: {self.exp_dir.name}")
        print(f"{'='*60}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Loss Configuration: {self.loss_manager}")
        print(f"{'='*60}\n")

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            self.current_epoch = epoch

            train_losses = self.train_epoch()
            if train_losses is None:
                break
            val_metrics = self.validate()
            if self.global_step >= self.max_steps:
                print(f"\n{'='*60}")
                print(f"Max steps ({self.max_steps}) reached. Training stopped.")
                print(f"{'='*60}")
                break
            # Get total losses
            train_loss_total = train_losses.get("total", 0)
            val_loss_total = val_metrics.get("total", 0) if val_metrics else None
            val_recon_loss = val_metrics.get("reconstruction", None)
            epoch_time = time.time() - epoch_start_time
            self.history["train_loss"].append(train_loss_total)
            if val_loss_total is not None:
                self.history["val_loss"].append(val_loss_total)
            self.history["epoch_times"].append(epoch_time)

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["learning_rates"].append(current_lr)

            # Log to TensorBoard - Training losses
            self.writer.add_scalar("Epoch/Train_Loss", train_loss_total, epoch)
            for loss_name, loss_value in train_losses.items():
                if loss_name != "total":
                    self.writer.add_scalar(
                        f"Epoch/Train_{loss_name}", loss_value, epoch
                    )

            # Log to TensorBoard - Validation losses
            if val_loss_total is not None:
                self.writer.add_scalar("Epoch/Val_Loss", val_loss_total, epoch)
                for loss_name, loss_value in val_metrics.items():
                    if loss_name != "total":
                        self.writer.add_scalar(
                            f"Epoch/Val_{loss_name}", loss_value, epoch
                        )
            self.writer.add_scalar("Epoch/Learning_Rate", current_lr, epoch)
            self.writer.add_scalar("Epoch/Time", epoch_time, epoch)
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs} Summary:")
            print(f"{'='*60}")
            print(f"   Train Loss (total): {train_loss_total:.6f}")
            for loss_name, loss_value in train_losses.items():
                if not loss_name.endswith("_weighted") and loss_name != "total":
                    print(f"      └─ {loss_name}: {loss_value:.6f}")
            if val_loss_total is not None:
                print(f"   Val Loss (total):   {val_loss_total:.6f}")
                for loss_name, loss_value in val_metrics.items():
                    if not loss_name.endswith("_weighted") and loss_name != "total":
                        print(f"      └─ {loss_name}: {loss_value:.6f}")

            print(f"   Time:               {epoch_time:.2f}s")
            print(f"   Learning Rate:      {current_lr:.2e}")

            # Check for best model (in terms of val loss)
            if val_recon_loss is not None:
                if val_recon_loss < self.best_val_loss:
                    self.best_val_loss = val_recon_loss
                    self.steps_without_improvement = 0
                    is_best = True
                    print("New best model (reconstruction loss)!")
                    self._save_checkpoint(epoch, is_best=is_best)
            # if (
            #    early_stopping_patience
            #    and self.epochs_without_improvement >= early_stopping_patience
            # ):
            #    print(f"\n{'='*60}")
            #    print(f" /!/ Early stopping triggered after {epoch+1} epochs /!/ ")
            #    print(f"     No improvement for {early_stopping_patience} epochs")
            #    print(f"{'='*60}")
            #    break

        # Save final logs
        self._save_logs()

        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}")
        print(f"Best Val Loss: {self.best_val_loss:.6f}")
        print(f"Total Epochs: {self.current_epoch + 1}")
        print(f"Logs saved to: {self.exp_dir}")
        print(f"\nTo view TensorBoard:")
        print(f"  tensorboard --logdir={self.exp_dir.parent}")
        print(f"{'='*60}\n")

        self.writer.close()

    def _save_logs(self):
        log_path = self.log_dir / "training_history.json"
        with open(log_path, "w") as f:
            json.dump(self.history, f, indent=4)
        print(f"Training history saved to {log_path}")

    def validate_and_check_early_stop(self, epoch):
        val_metrics = self.validate()
        if val_metrics is None:
            return False

        val_recon = val_metrics.get("reconstruction", None)
        is_best = False
        print(self.steps_without_improvement)
        if val_recon is not None:
            if val_recon < self.best_val_loss:
                self.best_val_loss = val_recon
                self.steps_without_improvement = 0
                is_best = True
                self._save_checkpoint(epoch, is_best=is_best)
                print(f"[Validation] New best recon loss: {val_recon:.6f}")
            else:
                self.steps_without_improvement += 1
                print(
                    f"[Validation] No improvement "
                    f"({self.steps_without_improvement}/"
                    f"{self.early_stopping_patience_steps})"
                )

        if (
            self.early_stopping_patience_steps
            and self.steps_without_improvement >= self.early_stopping_patience_steps
        ):
            print("\n/!/ Early stopping triggered (step-based) /!/")
            return True

        return False
