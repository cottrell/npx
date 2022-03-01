# from __future__ import annotations
import numpy as np
# from numpy.typing import ArrayLike


def _unique_tol(unique_fun, a, tol, **kwargs):
    a = np.asarray(a)
    aint = (a * (1.0 / tol)).astype(int)
    return_index = kwargs.pop('return_index', False)
    _, idx, *out = unique_fun(aint, return_index=True, **kwargs)
    unique_a = a[idx]
    if return_index:
        out = [idx, *out]
    if len(out) == 0:
        return unique_a
    return unique_a, *out


def unique(a, tol=0.0, **kwargs):
    assert tol >= 0.0
    if tol > 0.0:
        return _unique_tol(np.unique, a, tol, **kwargs)
    return np.unique(a, **kwargs)


def unique_rows(a, return_index=False, return_inverse=False, return_counts=
    False):
    a = np.asarray(a)
    a_shape = a.shape
    a = a.reshape(a.shape[0], np.prod(a.shape[1:], dtype=int))
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize *
        a.shape[1])))
    out = np.unique(b, return_index=return_index, return_inverse=
        return_inverse, return_counts=return_counts)
    if isinstance(out, tuple):
        out = out[0].view(a.dtype).reshape(out[0].shape[0], *a_shape[1:]
            ), *out[1:]
    else:
        out = out.view(a.dtype).reshape(out.shape[0], *a_shape[1:])
    return out
