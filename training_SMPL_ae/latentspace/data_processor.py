# data_processor.py
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import multiprocessing as mp
import traceback


# UMAP
def _umap_worker(latents_np_scaled, queue):
    try:
        import umap

        reducer = umap.UMAP(n_neighbors=50, n_components=2, random_state=42)
        coords = reducer.fit_transform(latents_np_scaled)
        queue.put(("ok", coords))
    except Exception as e:
        queue.put(("err", traceback.format_exc()))


def run_umap_subprocess(latents: torch.Tensor, timeout=600):
    latents_np = latents.numpy()
    latents_scaled = StandardScaler().fit_transform(latents_np)

    queue = mp.Queue()
    p = mp.Process(target=_umap_worker, args=(latents_scaled, queue))
    p.start()

    status, payload = queue.get(timeout=timeout)
    p.join()

    if status == "ok":
        return payload
    else:
        raise RuntimeError(f"UMAP subprocess error:\n{payload}")


# PCA & T-SNE
class DataProcessor:
    @staticmethod
    def reduce_pca_tsne(latents: torch.Tensor, perplexity=30):
        latents_np = latents.numpy()
        pca = PCA(n_components=2).fit_transform(latents_np)
        tsne = TSNE(
            n_components=2, random_state=42, perplexity=perplexity
        ).fit_transform(latents_np)
        return pca, tsne
