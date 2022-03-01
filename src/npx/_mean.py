import numpy as np
# from numpy.typing import ArrayLike


def _logsumexp(x):
    c = np.max(x)
    return c + np.log(np.sum(np.exp(x - c)))


def mean(x, p=1):
    """Generalized mean.



    See <https://github.com/numpy/numpy/issues/19341> for the numpy issue.

    """
    x = np.asarray(x)
    n = len(x)
    if p == 1:
        return np.mean(x)
    if p == -np.inf:
        return np.min(np.abs(x))
    elif p == 0:
        if np.any(x < 0.0):
            raise ValueError('p=0 only works with nonnegative x.')
        return np.prod(np.power(x, 1 / n))
    elif p == np.inf:
        return np.max(np.abs(x))
    if np.all(x > 0.0):
        return np.exp((_logsumexp(p * np.log(x)) - np.log(n)) / p)
    if not isinstance(p, (int, np.integer)):
        raise ValueError(f'Non-integer p (={p}) only work with nonnegative x.')
    return (np.sum(x ** p) / n) ** (1.0 / p)
