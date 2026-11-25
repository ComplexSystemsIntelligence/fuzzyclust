# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Data generators for the co-clustering examples."""

from __future__ import annotations

import numpy as np
from sklearn.utils import check_random_state

__all__ = ["make_block_matrix"]


def make_block_matrix(
    n_blocks: int = 3,
    n_objects_per_block=40,
    n_items_per_block=20,
    in_rate: float = 3.0,
    out_rate: float = 0.3,
    n_noise_items: int = 0,
    shuffle: bool = True,
    random_state=None,
):
    r"""A block-structured object--item count matrix with known co-clusters.

    Objects and items are grouped into ``n_blocks`` co-clusters; an entry
    is a Poisson count with mean ``in_rate`` when object and item belong
    to the same block and ``out_rate`` otherwise (a document--term matrix
    with topics and their vocabularies). ``n_noise_items`` extra items
    cooccur with every block at the background rate. Rows and columns are
    shuffled by default so that the block structure is hidden from the
    eye and has to be recovered.

    Parameters
    ----------
    n_blocks : int, default=3
    n_objects_per_block : int or sequence of int, default=40
    n_items_per_block : int or sequence of int, default=20
    in_rate : float, default=3.0
        Poisson mean inside the blocks.
    out_rate : float, default=0.3
        Poisson mean outside the blocks.
    n_noise_items : int, default=0
        Items without a block (label ``-1``).
    shuffle : bool, default=True
    random_state : int, RandomState instance or None

    Returns
    -------
    R : ndarray of shape (n_objects, n_items)
        Count matrix.
    row_labels : ndarray of shape (n_objects,)
        Block of every object.
    col_labels : ndarray of shape (n_items,)
        Block of every item (``-1`` for noise items).

    Examples
    --------
    >>> from fuzzyclust.coclustering import make_block_matrix
    >>> R, rows, cols = make_block_matrix(n_blocks=2, n_objects_per_block=10, n_items_per_block=5,
    ...                                   random_state=0)
    >>> R.shape, int(rows.max()), int(cols.max())
    ((20, 10), 1, 1)
    """
    rng = check_random_state(random_state)
    n_obj = np.broadcast_to(np.asarray(n_objects_per_block, dtype=int), (n_blocks,))
    n_it = np.broadcast_to(np.asarray(n_items_per_block, dtype=int), (n_blocks,))
    row_labels = np.repeat(np.arange(n_blocks), n_obj)
    col_labels = np.r_[np.repeat(np.arange(n_blocks), n_it), np.full(int(n_noise_items), -1)]
    same = row_labels[:, None] == col_labels[None, :]
    rate = np.where(same, in_rate, out_rate)
    R = rng.poisson(rate).astype(float)
    if shuffle:
        ri = rng.permutation(R.shape[0])
        ci = rng.permutation(R.shape[1])
        R = R[ri][:, ci]
        row_labels = row_labels[ri]
        col_labels = col_labels[ci]
    return R, row_labels, col_labels
