import numpy as np

# FIXME decorator??
def _check_y_score(y):
    return isinstance(y, np.ndarray) and len(y.shape) == 3

def compute_variation_ratio(y_score):
    assert _check_y_score(y_score)

    is_mode = y_score == y_score.max(axis=2, keepdims=True)
    is_mode = is_mode.astype(y_score.dtype)

    num_mode = is_mode.sum(axis=2, keepdims=True)

    mode_frequency = (is_mode / num_mode).sum(axis=0).max(axis=1)

    variation_ratio = 1 - (mode_frequency / len(y_score))
    # return np.clip(variation_ratio, 0, 1)
    return variation_ratio


def compute_predictive_entropy(y_score, eps=1e-7):
    '''
    Args:
        y_score: numpy.ndarray of shape (num_samples, batch size, num_classes)
    Returns:
    '''
    assert _check_y_score(y_score)
    # (T, B, C) --> (B, C)
    p_c = y_score.mean(axis=0)

    safe_p_c = np.clip(p_c, eps, 1.0)
    log_p_c = np.log(safe_p_c)

    p_log_p = p_c * log_p_c
    predictive_entropy = -p_log_p.sum(axis=1)
    return predictive_entropy


def compute_mutual_information(y_score, eps=1e-7):
    '''
    y: (T, B, C)
    '''
    assert _check_y_score(y_score)

    H = compute_predictive_entropy(y_score)

    log_y = np.log(np.clip(y_score, eps, 1))
    E_p_log_p = (y_score * log_y).sum(axis=(0, 2)) / len(y_score)
    return H + E_p_log_p
