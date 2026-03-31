import torch
import torch.nn as nn

from .encdec import Encoder, Decoder
from .loss import KL_Loss


class TemporalAttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Conv1d(dim, 1, 1)

    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=-1)
        return (x * w).sum(dim=2)


class VAE(nn.Module):
    def __init__(self, config):
        super().__init__()

        input_dim = config.get("input_dim", 263)
        latent_dim = config.get("latent_dim", 512)
        down_t = config.get("down_t", 3)
        stride_t = config.get("stride_t", 2)
        width = config.get("width", 512)
        depth = config.get("depth", 3)
        dilation_growth_rate = config.get("dilation_growth_rate", 3)
        activation = config.get("activation", "relu")
        norm = config.get("norm", None)
        kl_weight = config.get("kl_weight", 1e-4)
        nll_loss_type = config.get("nll_loss_type", "l1")

        self.down_t = down_t
        self.stride_t = stride_t
        self.latent_dim = latent_dim

        self.encoder = Encoder(
            input_dim,
            latent_dim,
            down_t,
            stride_t,
            width,
            depth,
            dilation_growth_rate,
            activation=activation,
            norm=norm,
        )

        self.decoder = Decoder(
            input_dim,
            latent_dim,
            down_t,
            stride_t,
            width,
            depth,
            dilation_growth_rate,
            activation=activation,
            norm=norm,
        )

        self.temporal_pool = TemporalAttentionPool(latent_dim)

        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)
        self.fc_post = nn.Linear(latent_dim, latent_dim)

        self.loss = KL_Loss(
            kl_weight=kl_weight,
            nll_loss_type=nll_loss_type,
        )

    def preprocess(self, x):
        return x.permute(0, 2, 1).float()

    def postprocess(self, x):
        return x.permute(0, 2, 1).float()

    def encode(self, x):
        x = self.preprocess(x)
        h = self.encoder(x)
        h = self.temporal_pool(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return mu, logvar, z

    def decode(self, z, T):
        B = z.shape[0]
        T_down = T // (self.stride_t**self.down_t)
        z = self.fc_post(z)
        z = z.unsqueeze(-1).repeat(1, 1, T_down)
        x = self.decoder(z)
        return self.postprocess(x)

    def forward(self, x):
        T = x.shape[1]
        mu, logvar, z = self.encode(x)
        recon = self.decode(z, T)
        return recon, mu, logvar
