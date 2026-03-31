import torch
import torch.nn.functional as F
import importlib
import torch.nn as nn

from .distributions import DiagonalGaussianDistribution
from .encdec import Encoder, Decoder
from .loss import KL_Loss


def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


class AutoencoderKL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Hyperparameters from config dict
        input_width = config.get("input_dim", 263)
        output_emb_width = config.get("latent_dim", 512)
        down_t = config.get("down_t", 3)
        stride_t = config.get("stride_t", 2)
        width = config.get("width", 512)
        depth = config.get("depth", 3)
        dilation_growth_rate = config.get("dilation_growth_rate", 3)
        activation = config.get("activation", "relu")
        norm = config.get("norm", None)
        nll_loss_type = config.get("nll_loss_type", "l1")
        kl_weight = config.get("kl_weight", 1.0)
        ckpt_path = config.get("ckpt_path", None)
        ignore_keys = config.get("ignore_keys", [])

        print("[Debug] Activation: ", activation)

        # Encoder / Decoder
        self.encoder = Encoder(
            input_width,
            output_emb_width,
            down_t,
            stride_t,
            width,
            depth,
            dilation_growth_rate,
            activation=activation,
            norm=norm,
        )
        self.decoder = Decoder(
            input_width,
            output_emb_width,
            down_t,
            stride_t,
            width,
            depth,
            dilation_growth_rate,
            activation=activation,
            norm=norm,
        )

        # Quantization convolutions
        self.quant_conv = torch.nn.Conv1d(output_emb_width, 2 * output_emb_width, 1)
        self.post_quant_conv = torch.nn.Conv1d(output_emb_width, output_emb_width, 1)

        # Loss
        self.loss = KL_Loss(kl_weight=kl_weight, nll_loss_type=nll_loss_type)
        self.output_emb_width = output_emb_width

        # Load checkpoint if provided
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def preprocess(self, x):
        # (B, T, D) -> (B, D, T)
        return x.permute(0, 2, 1).float()

    def postprocess(self, x):
        # (B, D, T) -> (B, T, D)
        return x.permute(0, 2, 1).float()

    def encode(self, x):
        x_in = self.preprocess(x)
        h = self.encoder(x_in)
        h = self.quant_conv(h)
        h = self.postprocess(h)
        posterior = DiagonalGaussianDistribution(h)
        return posterior

    def decode(self, z):
        """
        Expect input of shape (B, T, emb_width)
        """
        z = self.preprocess(z)
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        dec = self.postprocess(dec)
        return dec

    def forward(self, input, sample_posterior=False, return_latent=False):
        posterior = self.encode(input)
        z = posterior.sample() if sample_posterior else posterior.mode()
        dec = self.decode(z)

        if return_latent:
            return dec, posterior, z
        return dec, posterior
