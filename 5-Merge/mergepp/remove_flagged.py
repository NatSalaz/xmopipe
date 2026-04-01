import numpy as np


def trim_flagged_borders(flagged: np.ndarray, max_gap: int = 10):
    """
    Cuts a prefix and suffix of "dirty" frames according to `flagged` *and* also removes in addition 'max_gap'
    frames before/after any True encountered.
    Returns (start, stop) : so we keep flagged[start:stop].
    """

    flagged = np.asarray(flagged, dtype=bool)
    n = len(flagged)

    # before
    any_true = False
    false_run = 0
    start = 0
    for i, f in enumerate(flagged):
        if f:
            any_true, false_run = True, 0
            start = i + 1  # just after True
        elif any_true:
            false_run += 1
            if false_run > max_gap:  # Too big of a gap
                start = i - false_run + 1
                break
            start = i + 1
        else:  # False before any True
            break

    # after
    any_true = False
    false_run = 0
    stop = n
    for j in range(n - 1, -1, -1):
        f = flagged[j]
        if f:
            any_true, false_run = True, 0
            stop = j  # directly on the True value
        elif any_true:
            false_run += 1
            if false_run > max_gap:
                stop = j + false_run
                break
            stop = j
        else:
            break

    # "buffer" around Trues
    if start:  # we did get frames to remove at the beginning
        start = min(start + int(max_gap / 2), n)
    if stop != n:  # we did get frames to remove at the end
        stop = max(stop - int(max_gap / 2), 0)

    # everything is removed if stop <= start
    if stop <= start:
        return n, 0

    return start, stop
