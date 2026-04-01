import numpy as np
from collections import Counter


def smoothEmotions(emotions, window_length):
    # Counts the number of emotions in a given window length and replace the emotions with the most counted ones
    tmp_emo = []
    for i in range(0, len(emotions) + 1):
        indice_debut = max(0, i - window_length)
        compteur = Counter(emotions[indice_debut:i])
        emo_count = list(compteur.values())
        emo_keys = list(compteur.keys())
        if emo_count:
            max_index = emo_count.index(max(emo_count))
            tmp_emo.append(emo_keys[max_index])
    tmp_emo = np.array(tmp_emo)
    return tmp_emo


def smoothZeros1D(data, window_length):
    data = np.array(data, dtype=float)
    result = data.copy()
    n = len(data)
    i = 0
    while i < n:
        if data[i] == 0:
            start = i
            while i < n and data[i] == 0:
                i += 1
            gap = i - start
            left_idx = start - 1 if start > 0 else None
            right_idx = i if i < n else None
            if left_idx is not None and right_idx is not None:
                left_val = data[left_idx]
                right_val = data[right_idx]
                if gap <= window_length:
                    result[start:i] = np.linspace(left_val, right_val, gap + 2)[1:-1]
                else:
                    result[start : start + window_length] = np.linspace(
                        left_val, 0, window_length + 1
                    )[1:]
                    result[i - window_length : i] = np.linspace(
                        0, right_val, window_length + 1
                    )[:-1]
            elif left_idx is not None:
                left_val = data[left_idx]
                steps = min(gap, window_length)
                result[start : start + steps] = np.linspace(left_val, 0, steps + 1)[1:]
            elif right_idx is not None:
                right_val = data[right_idx]
                steps = min(gap, window_length)
                result[i - steps : i] = np.linspace(0, right_val, steps + 1)[:-1]
        else:
            i += 1
    return result


def smoothZeros2D_withIslandRemoval(data2D, window_length):
    """
    We smooth 2D data and if there are a few values surrounded by zeros we get rid of them.
    """
    # Keep the original data to detect islands
    orig = np.array(data2D, dtype=float)
    # Smooth only the zeros column by column, leaving the non-zero values intact
    islrem = orig.copy()
    n = orig.shape[0]
    i = 0
    while i < n:
        # we consider a row as a candidate for being part of an island
        # if at least one value is non-zero in the original data.
        if np.any(orig[i] != 0):
            start = i
            while i < n and np.any(orig[i] != 0):
                i += 1
            end = i - 1
            # Remove the island only if it is surrounded (both preceding AND following) by rows entirely composed of 0.
            if start > 0 and i < n:
                if np.all(orig[start - 1] == 0) and np.all(orig[i] == 0):
                    # If the island (the sequence of non-zero rows) is smaller than window_length,
                    # remove it by setting ALL its values to 0.
                    if (end - start + 1) < window_length:
                        islrem[start : end + 1, :] = 0
        else:
            
            i += 1
        result = np.apply_along_axis(
            smoothZeros1D, 0, islrem, window_length=window_length
        )
    return result


def remove_islands(data2D, window_length):
    """
    Supprime les îlots de valeurs non nulles entourés de 0,
    si leur longueur est inférieure à window_length.
    """
    orig = np.array(data2D, dtype=float)
    islrem = orig.copy()
    n = orig.shape[0]
    i = 0
    while i < n:
        # If the row has at least one non-zero value
        if np.any(orig[i] != 0):
            start = i
            # Advance while the row has at least one non-zero value
            while i < n and np.any(orig[i] != 0):
                i += 1
            end = i - 1
            # Check that the island is surrounded by all-zero rows
            if start > 0 and i < n:
                if np.all(orig[start - 1] == 0) and np.all(orig[i] == 0):
                    # If the island is smaller than window_length, remove it
                    if (end - start + 1) < window_length:
                        islrem[start : end + 1, :] = 0
        else:
            i += 1
    return islrem


import numpy as np
from scipy.ndimage import gaussian_filter1d


def smooth2D_gaussian(data2D, window_length, sigma):
    """
    First removes islands then applies a Gaussian filter
    along the time axis (each column separately).

    Parameters:
      data2D : np.array
          2D array with time on axis 0.
      window_length : int
          Parameter for island removal.
      sigma : float
          Standard deviation of the Gaussian for smoothing.

    Returns:
      np.array : Smoothed array.
    """
    # Step 1: island removal
    data_clean = remove_islands(data2D, window_length)

    # Step 2: Gaussian filter along axis 0 (time)
    smoothed = gaussian_filter1d(data_clean, sigma=sigma, axis=0)
    return smoothed


def smooth2D_moving_average(data2D, window_length):
    """
    First removes islands then applies a moving average
    along the time axis (each column separately).

    Parameters:
      data2D : np.array
          2D array with time on axis 0.
      window_length : int
          Window size for island removal and smoothing.

    Returns:
      np.array : Smoothed array.
    """
    # Step 1: island removal
    data_clean = remove_islands(data2D, window_length)

    # Step 2: moving average per column
    kernel = np.ones(window_length) / window_length
    smoothed = np.apply_along_axis(
        lambda x: np.convolve(x, kernel, mode="same"), axis=0, arr=data_clean
    )
    return smoothed
