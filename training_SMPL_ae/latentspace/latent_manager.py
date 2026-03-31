import numpy as np
import torch
from models.vae.vae import VAE
from models.vqvae.vqvae import VQVAE
from models.rvqvae.rvqvae import RVQVAE
from models.klvae.autoencoder import AutoencoderKL
from data.motion_loader import MotionDataset
from os.path import join as pjoin


class LatentManager:
    """Manages VAE model and latent encoding/decoding"""

    def __init__(self, device: torch.device, config: dict):
        self.device = device
        self.config = config
        self.model = None
        self.model_mean = None  # Stats du dataset d'entraînement du modèle
        self.model_std = None
        self.input_mean = None  # Stats du dataset d'entrée
        self.input_std = None
        self.dataloader = None

    def load_model(self, model_name: str, config: dict):
        """Load VAE model from checkpoint"""
        input_dim = config["input_dim"]
        latent_dim = config["latent_dim"]

        if model_name == "vae":
            self.model = VAE(config=config)
        elif model_name == "vqvae":
            self.model = VQVAE(config)
        elif model_name == "rvqvae":
            self.model = RVQVAE(config=config)
        elif model_name == "klvae":
            self.model = AutoencoderKL(config=config)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Use checkpoint from config if provided, otherwise use default path
        if "checkpoint" in config and config["checkpoint"] is not None:
            ckpt_path = config["checkpoint"]
        else:
            ckpt_path = f"experiments/{config['model_type']}_{config['dataset_name']}/checkpoints/latest.pth"

        print("Checkpoint taken from:", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        print(self.model)
        print("Model weights loaded successfully.")

        # Charger les statistiques du dataset d'entraînement du modèle
        data_trained_dir = self.config["data_root"]
        if data_trained_dir is None:
            data_trained_dir = MotionDataset(
                dataset_name=self.config["dataset_name"], subsampling=0.0
            ).data_root

        self.model_mean = np.load(pjoin(f"{data_trained_dir}/Mean.npy"))
        self.model_std = np.load(pjoin(f"{data_trained_dir}/Std.npy"))
        print(f"Loaded model training stats from: {data_trained_dir}")

    def encode_dataset(self, dataloader):
        """Encode a dataset that may have different normalization than training data"""
        self.dataloader = dataloader

        # Sauvegarder les stats du dataset d'entrée
        self.input_mean = dataloader.dataset.mean
        self.input_std = dataloader.dataset.std

        print(f"Input dataset std[150]: {self.input_std[150]}")
        print(f"Model training std[150]: {self.model_std[150]}")

        all_z_e_full = []
        all_z_q_full = []

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                x = batch.float().to(self.device)

                # Conversion: données normalisées input -> normalisées modèle
                # Étape 1: Dénormaliser avec les stats du dataset d'entrée
                x_np = x.cpu().numpy()
                # x_denorm_np = x_np * self.input_std + self.input_mean

                # Étape 2: Renormaliser avec les stats du modèle d'entraînement
                x_renorm_np = (x_np - self.model_mean) / self.model_std
                x_tensor = torch.from_numpy(x_renorm_np).float().to(self.device)

                # Encoding selon le type de modèle
                if self.config["model_type"] == "vqvae":
                    code_idx = self.model.encode(x_tensor)
                    all_z_q_full.append(code_idx.cpu())

                elif self.config["model_type"] == "rvqvae":
                    code_idx, all_codes = self.model.encode(x_tensor)
                    all_z_q_full.append(code_idx.cpu())
                    latent_mean = all_codes.mean(dim=0)
                    latent_flat = latent_mean.reshape(latent_mean.shape[0], -1)
                    all_z_e_full.append(latent_flat.cpu())

                elif self.config["model_type"] == "klvae":
                    z = self.model.encode(x_tensor).mode()
                    all_z_e_full.append(z.cpu())
                    all_z_q_full.append(z.cpu())
                else:
                    # VAE
                    z = self.model.encode(x_tensor)[0]
                    all_z_e_full.append(z.cpu())
                    all_z_q_full.append(z.cpu())

        if self.config["model_type"] == "vqvae":
            codebook_latent = self.model.quantizer.codebook.detach().cpu()
            all_z_e_full.append(codebook_latent)

        all_z_e_full = torch.cat(all_z_e_full, dim=0)
        all_z_q_full = torch.cat(all_z_q_full, dim=0)
        return all_z_e_full, all_z_q_full

    def decode(self, z_full, return_in_input_space=True):
        """
        Decode latent vectors back to motion space

        Args:
            z_full: latent vectors
            return_in_input_space: if True, return data normalized with input dataset stats
                                  if False, return data normalized with model training stats
        """
        # print("Decoding vector of size ", z_full.shape)
        self.model.eval()
        with torch.no_grad():
            if self.config["model_type"] == "vae":
                z_full = z_full.to(self.device)
                recon = self.model.decode(z_full, 64)
            elif self.config["model_type"] == "vqvae":
                if z_full.dim() == 2:
                    z_full = z_full.view(1, -1, self.config["token_length"])
                else:
                    z_full = z_full.long()

                z_full = z_full.to(self.device)
                recon = self.model.decode(z_full)
                recon = recon.view(self.config["window_size"], -1)

            elif self.config["model_type"] == "rvqvae":
                if z_full.dim() == 2:
                    z_full = z_full.unsqueeze(0)

                z_full = z_full.to(self.device)
                recon = self.model.forward_decoder(z_full)

            else:
                z_full = z_full.to(self.device)
                recon = self.model.decode(z_full)

            # Le modèle décode dans l'espace normalisé du dataset d'entraînement
            # Dénormaliser avec les stats du modèle
            recon_np = recon.cpu().numpy() * self.model_std + self.model_mean

            # Si on veut retourner dans l'espace du dataset d'entrée
            if return_in_input_space and self.input_mean is not None:
                # Renormaliser avec les stats du dataset d'entrée
                recon_np = (recon_np - self.input_mean) / self.input_std

            recon = torch.from_numpy(recon_np).float().to(self.device)

        return recon
