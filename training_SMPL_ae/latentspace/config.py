"""config.py - Configuration management"""

import yaml


def load_config(config_path: str) -> dict:
    """Load config from YAML"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


class ViewerConfig:
    """Default viewer configuration"""

    DATASET_NAME = "idea400"
    DATASET_TO_LOAD_NAME = "idea400"
    BATCH_SIZE = 32
    SUBSAMPLE_STEP = 10
    TSNE_PERPLEXITY = 30
    N_NEIGHBORS_UMAP = 30
    SUBSAMPLING_DATA = 0.1
