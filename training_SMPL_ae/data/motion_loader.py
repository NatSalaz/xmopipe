import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
from pathlib import Path
import random
import codecs as cs
import yaml
from tqdm import tqdm
from torch.utils.data import Subset


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_dataset_registry():
    """Read the dataset locations from the `training` section of config.yml."""
    with open(REPO_ROOT / "config.yml", "r") as f:
        cfg = yaml.safe_load(f)["training"]
    return (REPO_ROOT / cfg["dataset_dir"]).resolve(), cfg["dataset_dirs"]


DATASET_DIR, DATASET_DIRS = _load_dataset_registry()


def dataset_root(dataset_name):
    """Absolute path of a dataset, from its name in configs/<dataset>/<model>.yaml"""
    try:
        entry = DATASET_DIRS[dataset_name]
    except KeyError:
        raise ValueError(
            f"Unknown dataset name: {dataset_name}. "
            f"Known: {sorted(DATASET_DIRS)}. "
            f"Add it under training.dataset_dirs in {REPO_ROOT / 'config.yml'}."
        )
    # An entry is either the subfolder, or {path: ..., texts: ...} when the
    # texts do not sit in a plain `texts/` subfolder (Idea400).
    subdir = entry["path"] if isinstance(entry, dict) else entry
    texts = entry.get("texts", "texts") if isinstance(entry, dict) else "texts"
    return str(DATASET_DIR / subdir), texts


class MotionDataset(data.Dataset):

    def __init__(
        self,
        dataset_name,
        window_size=64,
        split="train",
        data_root=None,
        subsampling=None,
        deterministic=False,
        normalized=True,
    ):
        self.window_size = window_size
        self.dataset_name = dataset_name
        self.subsampling = subsampling
        self.deterministic = deterministic
        self.split = split
        self.normalized = normalized
        # `data_root` in the config wins over the registry in config.yml
        if data_root is not None:
            self.data_root, text_subdir = data_root, "texts"
        else:
            self.data_root, text_subdir = dataset_root(dataset_name)

        self.motion_dir = pjoin(self.data_root, "new_joint_vecs")
        self.text_dir = pjoin(self.data_root, text_subdir)
        self.joints_num = 22

        if not Path(self.motion_dir).is_dir():
            raise FileNotFoundError(
                f"No new_joint_vecs/ under {self.data_root} (dataset '{dataset_name}'). "
                f"Check training.dataset_dir / training.dataset_dirs in "
                f"{REPO_ROOT / 'config.yml'}."
            )

        # print("mean and std taken from", self.data_root)
        mean = np.load(pjoin(self.data_root, "Mean.npy"))
        std = np.load(pjoin(self.data_root, "Std.npy"))
        split_file = pjoin(self.data_root, f"{self.split}.txt")

        self.data = []
        # if "xmo" in self.dataset_name:
        self.motion_files = []
        self.lengths = []
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        if self.subsampling is not None:
            N = len(id_list)
            subset_len = max(1, int(N * self.subsampling))
            id_list = np.random.choice(id_list, subset_len, replace=False)

        for name in tqdm(id_list, desc=f"Loading {self.split} set", disable=True):
            try:
                motion = np.load(pjoin(self.motion_dir, name + ".npy"))

                if motion.shape[0] < self.window_size:
                    continue
                if np.isnan(motion).any():
                    continue
                # if "xmo" in self.dataset_name:
                self.motion_files.append(f"{self.motion_dir}/{name}.npy")
                #    self.lengths.append(motion.shape[0] - self.window_size)
                # else:
                self.lengths.append(motion.shape[0] - self.window_size)
                # self.data.append(motion)
            except:
                pass
        self.mean = mean
        self.std = std

    def inv_transform(self, data):
        return data * self.std + self.mean

    def compute_sampling_prob(self):
        prob = np.array(self.lengths, dtype=np.float32)
        prob /= np.sum(prob)
        return prob

    def __len__(self):
        # if "xmo" in self.dataset_name:
        return len(self.motion_files)

    # else:
    #    return len(self.data)

    def __getitem__(self, idx):
        # if "xmo" in self.dataset_name:
        name = self.motion_files[idx]
        motion = np.load(pjoin(name))
        # else:
        #    motion = self.data[idx]
        if not self.deterministic:
            start_idx = random.randint(0, len(motion) - self.window_size)
        else:
            rng = random.Random(int(idx))
            start_idx = rng.randint(0, len(motion) - self.window_size)
        motion_window = motion[start_idx : start_idx + self.window_size]
        if self.normalized:
            motion_window = (motion_window - self.mean) / self.std
        return torch.from_numpy(motion_window).float()


def DATALoader(
    dataset_name,
    batch_size,
    num_workers=8,
    window_size=64,
    data_root=None,
    shuffle=True,
    subsampling=None,
    deterministic=False,
    split=True,
    data_split="",
    normalized=True,
):
    if data_split == "":
        if split:
            train_dataset = MotionDataset(
                dataset_name,
                split="train",
                window_size=window_size,
                data_root=data_root,
                subsampling=subsampling,
                deterministic=deterministic,
                normalized=normalized,
            )

            val_dataset = MotionDataset(
                dataset_name,
                split="val",
                window_size=window_size,
                data_root=data_root,
                subsampling=subsampling,
                deterministic=True,
                normalized=normalized,
            )

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                drop_last=True,
            )

            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=True,
            )

            return train_loader, val_loader
        else:
            full_dataset = MotionDataset(
                dataset_name,
                split="all",
                window_size=window_size,
                data_root=data_root,
                subsampling=subsampling,
                deterministic=deterministic,
                normalized=normalized,
            )
            full_loader = torch.utils.data.DataLoader(
                full_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                drop_last=False,
            )
            return full_loader
    else:
        full_dataset = MotionDataset(
            dataset_name,
            split=data_split,
            window_size=window_size,
            data_root=data_root,
            subsampling=subsampling,
            deterministic=deterministic,
            normalized=normalized,
        )
        full_loader = torch.utils.data.DataLoader(
            full_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=False,
        )
        return full_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x
