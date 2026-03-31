# Code taken and lightly modified from https://github.com/Mael-zys/T2M-GPT/blob/main/models

import torch
import torch.nn as nn
import torch.nn.functional as F
from .encdec import Encoder, Decoder
from .quantize_cnn import QuantizeEMAReset


class VQVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.nb_joints = 22
        input_dim = config["input_dim"]

        # Encoder / Decoder
        self.encoder = Encoder(
            input_emb_width=input_dim,
            output_emb_width=config.get("hidden_dims", [512])[-1],
            down_t=config.get("down_t", 3),
            stride_t=config.get("stride_t", 2),
            width=config.get("hidden_dims", [512])[0],
            depth=config.get("depth", 3),
            dilation_growth_rate=config.get("dilation_growth_rate", 3),
            activation=config.get("activation", "relu"),
            norm=config.get("norm", None),
        )
        self.decoder = Decoder(
            input_emb_width=input_dim,
            output_emb_width=config.get("hidden_dims", [512])[-1],
            down_t=config.get("down_t", 3),
            stride_t=config.get("stride_t", 2),
            width=config.get("hidden_dims", [512])[0],
            depth=config.get("depth", 3),
            dilation_growth_rate=config.get("dilation_growth_rate", 3),
            activation=config.get("activation", "relu"),
            norm=config.get("norm", None),
        )

        # Quantizer EMA
        self.quantizer = QuantizeEMAReset(
            nb_code=config.get("num_embeddings", 512),
            code_dim=config.get("latent_dim", 256),
            args=config,
        )

        self.code_dim = config.get("latent_dim", 256)

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def encode(self, x):
        B, T, D = x.shape
        chunk_size = self.config["token_length"]
        assert T % chunk_size == 0
        num_chunks = T // chunk_size
        x_chunks = x.view(B * num_chunks, chunk_size, D).permute(0, 2, 1).contiguous()
        x_encoder = self.encoder(x_chunks)
        x_encoder = x_encoder.permute(0, 2, 1).contiguous()
        x_encoder = x_encoder.view(-1, x_encoder.shape[-1])
        code_idx = self.quantizer.quantize(x_encoder)
        code_idx = code_idx.view(B, num_chunks, -1)
        return code_idx

    def decode(self, code_idx):
        B, num_chunks, _ = code_idx.shape
        chunk_size = self.config["token_length"]
        x_d = self.quantizer.dequantize(code_idx.view(-1, code_idx.shape[-1]))
        x_d = (
            x_d.view(B * num_chunks, chunk_size, self.code_dim)
            .permute(0, 2, 1)
            .contiguous()
        )
        x_decoder = self.decoder(x_d)
        x_out = x_decoder.permute(0, 2, 1).contiguous()
        x_out = x_out.view(B, num_chunks * chunk_size, -1)
        return x_out

    def forward(self, x):
        B, T, D = x.shape
        chunk_size = self.config["token_length"]
        assert T % chunk_size == 0
        num_chunks = T // chunk_size
        x_chunks = x.view(B * num_chunks, chunk_size, D).permute(0, 2, 1).contiguous()
        x_encoder = self.encoder(x_chunks)
        x_quantized, loss, perplexity = self.quantizer(x_encoder)
        x_decoder = self.decoder(x_quantized)
        x_out = x_decoder.permute(0, 2, 1).contiguous()
        x_out = x_out.view(B, num_chunks * chunk_size, -1)
        return x_out, loss, perplexity

    """def preprocess(self, x):
        return x.permute(0, 2, 1).contiguous()

    def postprocess(self, x):
        return x.permute(0, 2, 1)

    def encode(self, x):
        x_enc = self.preprocess(x)
        
        x_enc = self.encoder(x_enc)
        N, C, T = x_enc.shape

        x_flat = x_enc.permute(0, 2, 1).contiguous().view(-1, C)
        
        code_idx = self.quantizer.quantize(x_flat)
        return code_idx.view(N, T)

    def forward(self, x):
        x_in = self.preprocess(x)
        x_enc = self.encoder(x_in)
        x_quantized, commit_loss, perplexity = self.quantizer(x_enc)
        x_dec = self.decoder(x_quantized)
        x_out = self.postprocess(x_dec)
        return x_out, commit_loss, perplexity

    def decode(self, code_idx):
        x_d = self.quantizer.dequantize(code_idx)
        x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        x_dec = self.decoder(x_d)
        x_out = self.postprocess(x_dec)
        return x_out"""
