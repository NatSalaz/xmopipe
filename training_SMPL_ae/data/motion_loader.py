import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
from torch.utils.data import Subset


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
        if dataset_name == "t2m":
            self.data_root = "/home/natsalaz/Documents/datasets/HumanML3D/HumanML3D"
            self.motion_dir = pjoin(self.data_root, "new_joint_vecs")
            self.text_dir = pjoin(self.data_root, "texts")
            self.joints_num = 22
            self.max_motion_length = 196
            self.meta_dir = "./checkpoints/t2m/kl_vae_ver0-stable/meta"

        elif dataset_name == "idea400":
            self.data_root = "/home/natsalaz/Documents/datasets/Idea400"
            self.motion_dir = pjoin(self.data_root, "new_joint_vecs")
            self.text_dir = pjoin(self.data_root, "idea400_txt")
            self.joints_num = 22
            self.max_motion_length = 1027
            self.meta_dir = "./checkpoints/t2m/kl_vae_ver0-stable/meta"

        elif dataset_name == "xmo":
            self.data_root = "/home/natsalaz/Documents/datasets/XmoPipe/XmoPipe"
            self.motion_dir = pjoin(self.data_root, "new_joint_vecs")
            self.text_dir = pjoin(self.data_root, "texts")
            self.joints_num = 22
            self.max_motion_length = 1027
            self.meta_dir = "./checkpoints/xmo/meta"

        elif dataset_name == "hml3dxmo":
            self.data_root = "/home/natsalaz/Documents/datasets/HML3Dxmo"
            self.motion_dir = pjoin(self.data_root, "new_joint_vecs")
            self.text_dir = pjoin(self.data_root, "texts")
            self.joints_num = 22
            self.max_motion_length = 1027
            self.meta_dir = "./checkpoints/hml3dxmo/meta"
        elif dataset_name == "xmoI400":
            self.data_root = "/home/natsalaz/Documents/datasets/XmoI400"
            self.motion_dir = pjoin(self.data_root, "new_joint_vecs")
            self.text_dir = pjoin(self.data_root, "texts")
            self.joints_num = 22
            self.max_motion_length = 1027
            self.meta_dir = "./checkpoints/xmoI400/meta"
        elif dataset_name == "hml3dxmoI400":
            self.data_root = "/home/natsalaz/Documents/datasets/HML3DxmoI400"
            self.motion_dir = pjoin(self.data_root, "new_joint_vecs")
            self.text_dir = pjoin(self.data_root, "texts")
            self.joints_num = 22
            self.max_motion_length = 1027
            self.meta_dir = "./checkpoints/hml3dxmoI400/meta"
        elif dataset_name == "hml3dI400":
            self.data_root = "/home/natsalaz/Documents/datasets/HML3DI400"
            self.motion_dir = pjoin(self.data_root, "new_joint_vecs")
            self.text_dir = pjoin(self.data_root, "texts")
            self.joints_num = 22
            self.max_motion_length = 1027
            self.meta_dir = "./checkpoints/hml3dxmoI400/meta"
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

        if data_root is not None:
            self.data_root = data_root
            self.motion_dir = pjoin(self.data_root, "new_joint_vecs")
            self.text_dir = pjoin(self.data_root, "texts")

        joints_num = self.joints_num

        # print("mean and std taken from", self.data_root)
        mean = np.load(pjoin(self.data_root, "Mean.npy"))
        std = np.load(pjoin(self.data_root, "Std.npy"))
        split_file = pjoin(self.data_root, f"{self.split}.txt")
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

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
