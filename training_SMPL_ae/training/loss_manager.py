import torch
import torch.nn as nn
from typing import Dict, Callable, Optional, Union, List
from .losses.recons_losses import ReConsLoss
from .losses.kl_loss import KL_Loss


class LossManager:
    """
    Multi-loss manager with weighting and detailed tracking.
    Preserves gradient flow by returning a single tensor for backward().

    Example:
        manager = LossManager()
        manager.add_loss("recon", nn.MSELoss(), weight=1.0)
        manager.add_loss("kl", lambda pred, target, mu, logvar: kl_divergence(mu, logvar), weight=0.5)

        total_loss, loss_dict = manager.compute(predictions=recon, targets=data, mu=mu, logvar=logvar)
        total_loss.backward()
    """

    def __init__(self):
        self.losses: Dict[str, Dict] = {}
        self.loss_history: Dict[str, List[float]] = {}

    def add_loss(
        self,
        name: str,
        loss_fn: Union[nn.Module, Callable],
        weight: float = 1.0,
        apply_to: Optional[List[str]] = None,
        active: bool = True,
    ):
        """
        Add a loss to the manager.

        Args:
            name: Loss identifier for logging
            loss_fn: Loss function or nn.Module
            weight: Multiplier for this loss in total sum
            apply_to: Argument names to extract from compute() kwargs
            active: Whether to compute this loss
        """
        self.losses[name] = {
            "fn": loss_fn,
            "weight": weight,
            "apply_to": apply_to or ["predictions", "targets"],
            "active": active,
        }
        self.loss_history[name] = []

    def remove_loss(self, name: str):
        """Remove a loss from the manager."""
        if name in self.losses:
            del self.losses[name]
            del self.loss_history[name]

    def set_weight(self, name: str, weight: float):
        """Update the weight of a loss."""
        if name in self.losses:
            self.losses[name]["weight"] = weight

    def activate(self, name: str, active: bool = True):
        """Enable or disable a loss."""
        if name in self.losses:
            self.losses[name]["active"] = active

    def compute(self, **kwargs) -> tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute all active losses.

        Args:
            **kwargs: Tensors required by loss functions (predictions, targets, mu, logvar, etc.)

        Returns:
            total_loss: Differentiable tensor for backward()
            loss_details: Dict of detached values for logging
        """
        total_loss = torch.tensor(0.0, device=self._get_device(kwargs))
        loss_details = {}

        for name, loss_info in self.losses.items():
            if not loss_info["active"]:
                continue

            loss_fn = loss_info["fn"]
            weight = loss_info["weight"]
            apply_to = loss_info["apply_to"]

            args = []
            for arg_name in apply_to:
                if arg_name not in kwargs:
                    raise ValueError(
                        f"Argument '{arg_name}' required for loss '{name}' "
                        f"but not provided in compute()"
                    )
                args.append(kwargs[arg_name])

            try:
                if isinstance(loss_fn, nn.Module):
                    loss_value = loss_fn(*args)
                else:
                    loss_value = loss_fn(*args)

                weighted_loss = weight * loss_value
                total_loss = total_loss + weighted_loss

                loss_details[name] = loss_value.detach().item()
                loss_details[f"{name}_weighted"] = weighted_loss.detach().item()

            except Exception as e:
                raise RuntimeError(f"Error computing loss '{name}': {e}")

        loss_details["total"] = total_loss.detach().item()

        for name, value in loss_details.items():
            if name not in self.loss_history:
                self.loss_history[name] = []
            self.loss_history[name].append(value)

        return total_loss, loss_details

    def _get_device(self, kwargs: dict) -> torch.device:
        """Detect device from provided tensors."""
        for v in kwargs.values():
            if isinstance(v, torch.Tensor):
                return v.device
        return torch.device("cpu")

    def get_history(self, name: Optional[str] = None) -> Dict[str, List[float]]:
        """Return loss history for one or all losses."""
        if name:
            return {name: self.loss_history.get(name, [])}
        return self.loss_history

    def reset_history(self):
        """Clear all loss history."""
        for name in self.loss_history:
            self.loss_history[name] = []

    def get_summary(self) -> Dict:
        """Return configuration summary of all losses."""
        return {
            name: {
                "weight": info["weight"],
                "active": info["active"],
                "apply_to": info["apply_to"],
            }
            for name, info in self.losses.items()
        }

    def __repr__(self):
        active_losses = [n for n, l in self.losses.items() if l["active"]]
        return f"LossManager( {len(active_losses)} active losses: {active_losses} )"


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL divergence for VAE."""
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def vq_loss(
    quantized: torch.Tensor, inputs: torch.Tensor, commitment: float = 0.25
) -> torch.Tensor:
    """VQ-VAE loss with commitment term."""
    e_latent_loss = torch.mean((quantized.detach() - inputs) ** 2)
    q_latent_loss = torch.mean((quantized - inputs.detach()) ** 2)
    return q_latent_loss + commitment * e_latent_loss


def perplexity(perplexity: torch.Tensor):
    """Pass-through for perplexity metric."""
    return perplexity


def create_vae_loss_manager(
    beta: float = 1.0, recon_weight: float = 1.0
) -> LossManager:
    """Create a LossManager configured for standard VAE training."""
    manager = LossManager()
    manager.add_loss(
        "reconstruction",
        ReConsLoss("l1_smooth", 22),
        # nn.MSELoss(),
        weight=1.0,
        apply_to=["predictions", "targets"],
    )
    manager.add_loss(
        "kl_divergence", kl_divergence, weight=beta, apply_to=["mu", "logvar"]
    )
    return manager


def create_klvae_loss_manager(
    beta: float = 0.001, recon_weight: float = 1.0, vel_weight: float = 0.5
) -> LossManager:
    """Create a LossManager with separate reconstruction and KL losses."""
    manager = LossManager()
    recon_loss = ReConsLoss("l1_smooth", 22)
    # Reconstruction loss séparée
    manager.add_loss(
        "reconstruction",
        recon_loss,
        weight=recon_weight,
        apply_to=["predictions", "targets"],
    )

    # KL loss seulement (extraire du posterior)
    manager.add_loss(
        "kl_divergence",
        lambda posterior: torch.sum(posterior.kl()) / posterior.kl().shape[0],
        weight=beta,
        apply_to=["posterior"],
    )

    # Velocity loss
    if vel_weight > 0:
        manager.add_loss(
            "velocity",
            lambda pred, target: recon_loss.forward_vel(pred, target),
            weight=vel_weight,
            apply_to=["predictions", "targets"],
        )

    return manager


def create_vqvae_loss_manager(
    vq_weight: float = 1.0, commitment: float = 0.25, recon_weight: float = 1.0
) -> LossManager:
    """Create a LossManager configured for VQ-VAE training."""
    manager = LossManager()
    manager.add_loss(
        "reconstruction",
        ReConsLoss("l1_smooth", 22),
        weight=1.0,
        apply_to=["predictions", "targets"],
    )
    manager.add_loss("perplexity", nn.Identity(), weight=0.0, apply_to=["perplexity"])
    manager.add_loss(
        "vq",
        lambda quantized, inputs: vq_loss(quantized, inputs, commitment),
        weight=vq_weight,
        apply_to=["quantized", "inputs"],
    )
    return manager


def create_rvqvae_loss_manager(
    vq_weight: float = 1.0, commitment: float = 0.25, recon_weight: float = 1.0
) -> LossManager:
    """Create a LossManager configured for VQ-VAE training."""
    manager = LossManager()
    manager.add_loss(
        "reconstruction",
        ReConsLoss("l1_smooth", 22),
        # nn.MSELoss(),
        weight=1.0,
        apply_to=["predictions", "targets"],
    )
    manager.add_loss("perplexity", nn.Identity(), weight=0.0, apply_to=["perplexity"])
    manager.add_loss(
        "vq",
        lambda quantized, inputs: vq_loss(quantized, inputs, commitment),
        weight=vq_weight,
        apply_to=["quantized", "inputs"],
    )
    return manager
