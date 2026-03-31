import torch
import yaml
import argparse
import numpy as np
from tqdm import tqdm
from scipy import linalg

from models.vae.vae import VAE
from models.vqvae.vqvae import VQVAE
from models.rvqvae.rvqvae import RVQVAE
from models.klvae.autoencoder import AutoencoderKL
from data.motion_loader import DATALoader, MotionDataset
from utils.motion_process import recover_from_ric
from utils.metric_utils import calc_mpjpe, calc_pampjpe, calc_accel
from models.t2m_eval_wrapper import EvaluatorModelWrapper
from utils.word_vectorizer import POS_enumerator
from os.path import join as pjoin
from latentspace.latent_manager import LatentManager


def get_model(model_name, config):
    """Factory to create model"""
    models = {
        "vae": lambda: VAE(config["input_dim"], config["latent_dim"]),
        "vqvae": lambda: VQVAE(config),
        "rvqvae": lambda: RVQVAE(config),
        "klvae": lambda: AutoencoderKL(config),
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    return models[model_name]()


def sample_latent(model, model_type, batch_size, seq_len, device, config):
    """
    Sample latent vectors for generation depending on model type
    """
    if model_type == "vae":
        z = torch.randn(batch_size, config["latent_dim"], device=device)
        return z

    elif model_type == "klvae":
        z = torch.randn(
            batch_size,
            config["z_channels"],
            seq_len // config.get("downsample_rate", 4),
            device=device,
        )
        return z

    elif model_type in ["vqvae", "rvqvae"]:
        # Uniform sampling in codebook
        num_codes = model.codebook_size
        z_q = torch.randint(0, num_codes, (batch_size, seq_len), device=device)
        return z_q

    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def generate_motion(model, model_type, z, device):
    """
    Decode latent samples into motion
    """
    if model_type == "vae":
        gen = model.decode(z)

    elif model_type == "klvae":
        gen = model.decode(z)

    elif model_type in ["vqvae", "rvqvae"]:
        gen = model.decode_from_indices(z)

    else:
        raise ValueError(model_type)

    return gen


def create_evaluator_options(dataset_name, device):
    """
    Create options object for EvaluatorModelWrapper from dataset name
    """
    from argparse import Namespace

    opt = Namespace()
    opt.dataset_name = dataset_name
    opt.device = device
    # Set dataset-specific parameters
    # print("this is the dataset", dataset_name)
    if dataset_name == "t2m":
        opt.data_root = "./dataset/HumanML3D/"
        opt.motion_dir = pjoin(opt.data_root, "new_joint_vecs")
        opt.text_dir = pjoin(opt.data_root, "texts")
        opt.joints_num = 22
        opt.dim_pose = 263
    elif dataset_name == "xmo":
        opt.data_root = "./dataset/XmoPipe/"
        opt.motion_dir = pjoin(opt.data_root, "new_joint_vecs")
        opt.text_dir = pjoin(opt.data_root, "texts")
        opt.joints_num = 22
        opt.dim_pose = 263
    elif dataset_name == "hml3dxmo":
        opt.data_root = "./dataset/HML3Dxmo/"
        opt.motion_dir = pjoin(opt.data_root, "new_joint_vecs")
        opt.text_dir = pjoin(opt.data_root, "texts")
        opt.joints_num = 22
        opt.dim_pose = 263
    elif dataset_name == "idea400":
        opt.data_root = "./dataset/Idea400/"
        opt.motion_dir = pjoin(opt.data_root, "new_joint_vecs")
        opt.text_dir = pjoin(opt.data_root, "texts")
        opt.joints_num = 22
        opt.dim_pose = 263
    elif dataset_name == "xmoI400":
        opt.data_root = "./dataset/xmoI400/"
        opt.motion_dir = pjoin(opt.data_root, "new_joint_vecs")
        opt.text_dir = pjoin(opt.data_root, "texts")
        opt.joints_num = 22
        opt.dim_pose = 263
    elif dataset_name == "hml3dI400":
        opt.data_root = "./dataset/HML3DI400/"
        opt.motion_dir = pjoin(opt.data_root, "new_joint_vecs")
        opt.text_dir = pjoin(opt.data_root, "texts")
        opt.joints_num = 22
        opt.dim_pose = 263
    elif dataset_name == "hml3dxmoI400":
        opt.data_root = "./dataset/HML3DxmoI400/"
        opt.motion_dir = pjoin(opt.data_root, "new_joint_vecs")
        opt.text_dir = pjoin(opt.data_root, "texts")
        opt.joints_num = 22
        opt.dim_pose = 263
    else:
        raise KeyError(f"Dataset {dataset_name} not recognized")

    # Common parameters for evaluator
    opt.unit_length = 4
    opt.max_motion_length = 64
    opt.max_motion_frame = 64
    opt.max_motion_token = 55
    opt.dim_word = 300
    opt.num_classes = 200 // opt.unit_length
    opt.dim_pos_ohot = len(POS_enumerator)
    opt.dim_text_hidden = 512
    opt.dim_coemb_hidden = 512
    opt.dim_motion_hidden = 1024
    opt.dim_movement_enc_hidden = 512
    opt.dim_movement_latent = 512
    opt.max_text_len = 20

    # Paths
    opt.checkpoints_dir = "./checkpoints"
    opt.name = "text_mot_match"
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, "model")
    opt.meta_dir = pjoin(opt.save_root, "meta")
    opt.which_epoch = "finest"

    opt.is_train = False
    opt.is_continue = False

    return opt


def calculate_activation_statistics(activations):
    """
    Calculate mean and covariance statistics for FID computation
    """
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma


def sample_latent_from_pool(Z_pool, batch_size):
    idx = torch.randint(0, Z_pool.shape[0], (batch_size,))
    return Z_pool[idx]


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Calculate Frechet Distance between two multivariate Gaussians
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_diversity(activation, diversity_times=300):
    """
    Calculate diversity score
    """
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times

    num_samples = activation.shape[0]
    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)

    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()


def evaluate_matching_score(eval_wrapper, motions, m_lens, texts, t_lens, device):
    """
    Evaluate motion-text matching score
    """
    with torch.no_grad():
        word_embs = texts[0].to(device)
        pos_ohot = texts[1].to(device)
        text_lengths = texts[2].to(device)

        text_emb, motion_emb = eval_wrapper.get_co_embeddings(
            word_embs, pos_ohot, text_lengths, motions, m_lens
        )

        dist_mat = euclidean_distance_matrix(
            text_emb.cpu().numpy(), motion_emb.cpu().numpy()
        )
        matching_score = dist_mat.trace()

    return matching_score


def euclidean_distance_matrix(matrix1, matrix2):
    """
    Compute euclidean distance matrix
    """
    matrix1_sq = np.sum(matrix1**2, axis=1, keepdims=True)
    matrix2_sq = np.sum(matrix2**2, axis=1, keepdims=True)

    dist_mat = matrix1_sq + matrix2_sq.T - 2 * np.dot(matrix1, matrix2.T)
    dist_mat = np.sqrt(np.maximum(dist_mat, 0))

    return dist_mat


def evaluate(
    config, checkpoint_path, batch_size=256, dataset_to_load=None, num_repeats=20
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config["seed"])
    if dataset_to_load is not None:
        config["dataset_name"] = dataset_to_load
    joints_num = 21 if config["dataset_name"] == "kit" else 22

    # Load model
    net = get_model(config["model_type"], config)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    net.load_state_dict(ckpt["model_state_dict"], strict=True)
    net.to(device).eval()

    if dataset_to_load is None:
        dataset_to_load = config["dataset_name"]

    # Load data
    loader = DATALoader(
        dataset_name=dataset_to_load,
        batch_size=batch_size,
        num_workers=config.get("num_workers", 4),
        window_size=config.get("window_size", 32),
        data_root=config.get("data_root"),
        shuffle=False,
        data_split="test",
        normalized=False,
    )

    data_trained_dir = config["data_root"]
    if data_trained_dir is None:
        data_trained = MotionDataset(
            dataset_name=config["dataset_name"], subsampling=0.0
        )

    mean = torch.from_numpy(data_trained.mean).to(device)
    std = torch.from_numpy(data_trained.std).to(device)

    # Setup evaluator for FID and matching scores
    # print("Loading evaluation wrapper for FID/Matching...")
    # print(config["dataset_name"])
    # wrapper_opt = create_evaluator_options(config["dataset_name"], device)
    # eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    # Initialize metric accumulators for reconstruction
    mpjpe_sum, pampjpe_sum, accl_sum = 0.0, 0.0, 0.0
    num_poses, num_accl = 0, 0

    # Lists for FID metrics (will compute over multiple runs)
    # all_fid = []
    # all_diversity = []
    # all_matching = []

    # print(f"\nEvaluating {config['model_type']} on {dataset_to_load}...")
    # print(f"Running {num_repeats} repetitions for statistical metrics...\n")

    # First pass: Compute reconstruction metrics (only once)
    # print("=" * 60)
    # print("COMPUTING RECONSTRUCTION METRICS (single pass)")
    # print("=" * 60)

    with torch.no_grad():
        for motions in tqdm(loader, desc="Reconstruction metrics"):
            motions = motions.to(device)
            motions_normalized = (motions - mean) / std
            B, T, _ = motions.shape

            # Forward pass
            if config["model_type"] in ["rvqvae", "vqvae"]:
                recon, _, _ = net(motions_normalized)
            elif config["model_type"] in ["klvae"]:
                z_dist = net.encode(motions_normalized).mode()
                recon = net.decode(z_dist)
            else:
                recon = net(motions_normalized)[0]

            # Denormalize and recover 3D joints
            gt_xyz = recover_from_ric(motions, joints_num)  # [B, T, J, 3]
            pred_xyz = recover_from_ric(recon * std + mean, joints_num)

            # Compute metrics per sample
            for i in range(B):
                gt, pred = gt_xyz[i], pred_xyz[i]  # [T, J, 3]

                # MPJPE
                mpjpe = calc_mpjpe(gt, pred)
                mpjpe_sum += mpjpe.sum().item()

                # PA-MPJPE
                pampjpe = calc_pampjpe(pred, gt)
                pampjpe_sum += pampjpe.sum().item()

                num_poses += T

                # Acceleration error
                accl_error = calc_accel(pred, gt)
                accl_sum += accl_error.mean().item()
                num_accl += 1

    recon_results = {
        "mpjpe": mpjpe_sum / num_poses,
        "pampjpe": pampjpe_sum / num_poses,
        "accl": accl_sum / num_accl if num_accl > 0 else 0.0,
    }

    # We won't calculate FID and Diversity since we work on 64 frames motions
    #    print("\nEncoding dataset to build latent pool...")
    #
    #    latent_manager = LatentManager(device, config)
    #    latent_manager.model = net
    #    latent_manager.model_mean = mean.cpu().numpy()
    #    latent_manager.model_std = std.cpu().numpy()
    #
    #    Z_e_full, Z_q_full = latent_manager.encode_dataset(loader)
    #
    #    print("Latent pool size:", Z_q_full.shape)
    #
    #    print("\n" + "=" * 60)
    #    print(f"COMPUTING FID & DIVERSITY ({num_repeats} generations)")
    #    print("=" * 60)
    #
    #    all_fid = []
    #    all_diversity = []
    #
    #    for repeat_idx in range(num_repeats):
    #        print(f"\nGeneration {repeat_idx + 1}/{num_repeats}")
    #
    #        real_embeddings = []
    #        gen_embeddings = []
    #
    #        with torch.no_grad():
    #            for motions in tqdm(loader, desc="FID sampling", leave=False):
    #                motions = motions.to(device)
    #                B, T, _ = motions.shape
    #                m_lens = torch.full((B,), T, device=device, dtype=torch.long)
    #
    #                # Real motions
    #                real_emb = eval_wrapper.get_motion_embeddings(motions, m_lens)
    #                real_embeddings.append(real_emb.cpu().numpy())
    #
    #                # Sample latent from real pool
    #                if config["model_type"] in ["vqvae", "rvqvae"]:
    #                    z = sample_latent_from_pool(Z_q_full, B)
    #                else:
    #                    z = sample_latent_from_pool(Z_e_full, B)
    #
    #                # Decode
    #                gen = latent_manager.decode(
    #                    z,
    #                    return_in_input_space=False
    #                )
    #
    #                gen_emb = eval_wrapper.get_motion_embeddings(gen, m_lens)
    #                gen_embeddings.append(gen_emb.cpu().numpy())
    #
    #        real_embeddings = np.concatenate(real_embeddings, axis=0)
    #        gen_embeddings = np.concatenate(gen_embeddings, axis=0)
    #
    #        # FID
    #        mu_r, sigma_r = calculate_activation_statistics(real_embeddings)
    #        mu_g, sigma_g = calculate_activation_statistics(gen_embeddings)
    #        fid = calculate_frechet_distance(mu_r, sigma_r, mu_g, sigma_g)
    #
    #        # Diversity
    #        diversity = calculate_diversity(gen_embeddings)
    #
    #        all_fid.append(fid)
    #        all_diversity.append(diversity)
    #
    #        print(f"  FID: {fid:.4f} | Diversity: {diversity:.4f}")
    #
    #
    #    all_fid = np.array(all_fid)
    #    all_diversity = np.array(all_diversity)

    # fid_mean, fid_std = all_fid.mean(), all_fid.std()
    # div_mean, div_std = all_diversity.mean(), all_diversity.std()

    # fid_conf = 1.96 * fid_std / np.sqrt(num_repeats)
    # div_conf = 1.96 * div_std / np.sqrt(num_repeats)

    print("=" * 80)
    print(
        f"Info: Model was trained during {ckpt['epoch']} epochs, {ckpt['global_step']} steps."
    )
    # print("=" * 80)

    print("RECONSTRUCTION METRICS (lower is better)")
    # print("-" * 80)
    print(f"  MPJPE:     {recon_results['mpjpe']*1000:.4f} mm")
    print(f"  PA-MPJPE:  {recon_results['pampjpe']*1000:.4f} mm")
    print(f"  ACCL:      {recon_results['accl']*1000:.4f} mm/s²")

    # print("\nPERCEPTUAL METRICS (mean ± 95% CI)")
    # print("-" * 80)
    # print(f"  FID:       {fid_mean:.3f} ± {fid_conf:.3f}")
    # print(f"  Diversity: {div_mean:.3f} ± {div_conf:.3f}")

    print("=" * 80 + "\n")

    results = {
        "reconstruction": recon_results,
        # "perceptual": {
        #    "fid_mean": fid_mean,
        #    "fid_conf": fid_conf,
        #    "diversity_mean": div_mean,
        #    "diversity_conf": div_conf,
        # }
    }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hybrid evaluation: Reconstruction + FID/Diversity/Matching"
    )
    parser.add_argument("--config", type=str, required=True, help="Config YAML file")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Model checkpoint"
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--dataset", type=str, help="Override dataset name")
    parser.add_argument(
        "--num-repeats",
        type=int,
        default=20,
        help="Number of repetitions for statistical metrics",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    results = evaluate(
        config, args.checkpoint, args.batch_size, args.dataset, args.num_repeats
    )
