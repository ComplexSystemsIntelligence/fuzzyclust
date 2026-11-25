# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Fuzzy c-means with partial supervision (Pedrycz and Waletzky, 1997)."""

from __future__ import annotations

import numpy as np
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, validate_data

from .._utils import EPS, fuzzy_covariances, memberships_from_distances, sq_euclidean
from ..base import BaseFuzzyClusterer

__all__ = ["SemiSupervisedFCM"]


class SemiSupervisedFCM(BaseFuzzyClusterer):
    r"""Fuzzy c-means with partial supervision (Pedrycz and Waletzky, 1997).

    A small subset of the samples carries a class label (or a fuzzy label
    vector). The c-means functional is extended by a term that measures the
    disagreement between the memberships of the *labeled* samples and
    their labels, weighted by :math:`\alpha`; the unlabeled samples are
    clustered as usual, and the labels steer the whole partition, because
    the prototypes are shared. In the scikit-learn convention of the
    semi-supervised module, ``y == -1`` marks an unlabeled sample.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters :math:`c`. Must be at least the number of
        classes in ``y``; class :math:`j` is tied to cluster :math:`j`
        (in the order of ``classes_``), surplus clusters are unsupervised.
    alpha : float, default=1.0
        Weight :math:`\alpha\ge0` of the supervision term. ``0`` gives plain
        FCM; the paper suggests :math:`\alpha` proportional to :math:`N/M`
        (number of samples over number of labeled samples), so that the
        labeled minority weighs as much as the unlabeled majority.
        ``alpha="auto"`` sets :math:`\alpha=N/M`.
    m : float, default=2.0
        Fuzziness exponent. The paper's closed-form update exists for
        :math:`m=2` only, which is the value implemented; another value
        raises ``ValueError``.
    tol : float, default=1e-4
    max_iter : int, default=300
    init : {"labels", "random", "kmeans++"} or ndarray, default="labels"
        ``"labels"`` places the prototypes of the supervised clusters at the
        means of their labeled samples (random prototypes for the surplus
        clusters and when no labels are given); the other options are those
        of :class:`~fuzzyclust.base.BaseFuzzyClusterer`. Since class
        :math:`j` is tied to cluster :math:`j`, a random start that puts
        cluster :math:`j` next to another class makes the supervision fight
        the geometry and ends in a poor local optimum; starting from the
        labels (or using ``n_init`` restarts, which keep the lowest
        supervised objective) avoids this.
    n_init : int, default=1
    random_state : int, RandomState instance or None, default=None
    verbose : int, default=0

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
    membership_ : ndarray of shape (n_samples, n_clusters)
    labels_ : ndarray of shape (n_samples,)
    classes_ : ndarray of shape (n_classes,)
        Sorted class labels seen in ``y`` (without ``-1``).
    label_mapping_ : ndarray of shape (n_clusters,)
        Class label tied to every cluster, ``-1`` for unsupervised clusters.
    label_membership_ : ndarray of shape (n_samples, n_clusters)
        The label matrix :math:`f_{ik}` used (zero rows for unlabeled samples).
    labelled_mask_ : ndarray of shape (n_samples,)
        Boolean indicator :math:`b_k` of the labeled samples.
    alpha_ : float
        The weight actually used.
    covariances_ : ndarray of shape (n_clusters, n_features, n_features)
        Fuzzy covariances of the final partition (for plotting).
    distances_, objective_, objective_history_, n_iter_
        See :class:`~fuzzyclust.base.BaseFuzzyClusterer`.

    Notes
    -----
    With the indicator :math:`b_k\in\{0,1\}` of the labeled samples and
    their label memberships :math:`f_{ik}` (:math:`\sum_i f_{ik}=1` for
    :math:`b_k=1`), the objective is

    .. math::
        J = \sum_{i=1}^{c}\sum_{k=1}^{N}\mu_{ik}^2 D_{ik}^2
          + \alpha\sum_{i=1}^{c}\sum_{k=1}^{N}(\mu_{ik}-f_{ik}b_k)^2 D_{ik}^2 .

    Its necessary conditions give the membership update

    .. math::
        \mu_{ik} = \frac{1}{1+\alpha}\left[
          \frac{1+\alpha\big(1-b_k\sum_{j=1}^{c}f_{jk}\big)}
               {\sum_{j=1}^{c} D_{ik}^2/D_{jk}^2} + \alpha f_{ik}b_k\right]

    and the prototype update

    .. math::
        \mathbf{v}_i = \frac{\sum_k\big[\mu_{ik}^2+\alpha(\mu_{ik}-f_{ik}b_k)^2\big]\mathbf{x}_k}
                            {\sum_k\big[\mu_{ik}^2+\alpha(\mu_{ik}-f_{ik}b_k)^2\big]} .

    For an unlabeled sample the bracket reduces to the FCM formula; for a
    labeled one the membership is a convex combination of the FCM
    membership and the label, with weight :math:`\alpha/(1+\alpha)` on the
    label. New samples (:meth:`predict_proba`) carry no label and receive
    FCM memberships to the supervised prototypes.

    References
    ----------
    .. [1] W. Pedrycz, J. Waletzky, "Fuzzy clustering with partial
           supervision", IEEE Transactions on Systems, Man, and
           Cybernetics, Part B 27(5), 787--795, 1997.

    Examples
    --------
    >>> import numpy as np
    >>> from fuzzyclust.semisupervised import SemiSupervisedFCM, make_partially_labeled_pair
    >>> X, y, y_partial = make_partially_labeled_pair(random_state=0)
    >>> ssfcm = SemiSupervisedFCM(n_clusters=2, alpha="auto", random_state=0).fit(X, y_partial)
    >>> ssfcm.membership_.shape, int((y_partial >= 0).sum()) == int(ssfcm.labelled_mask_.sum())
    ((300, 2), True)
    """

    def __init__(
        self,
        n_clusters: int = 3,
        alpha=1.0,
        m: float = 2.0,
        tol: float = 1e-4,
        max_iter: int = 300,
        init="labels",
        n_init: int = 1,
        random_state=None,
        verbose: int = 0,
    ):
        super().__init__(
            n_clusters=n_clusters,
            m=m,
            tol=tol,
            max_iter=max_iter,
            init=init,
            n_init=n_init,
            random_state=random_state,
            verbose=verbose,
        )
        self.alpha = alpha

    # ------------------------------------------------------------------ labels
    def _check_params(self, X):
        super()._check_params(X)
        if self.m != 2.0:
            raise ValueError("SemiSupervisedFCM implements the closed-form update of the paper, m must be 2.")
        if isinstance(self.alpha, str):
            if self.alpha != "auto":
                raise ValueError("alpha must be a non-negative float or 'auto'.")
        elif self.alpha < 0:
            raise ValueError("alpha must be non-negative.")

    def _label_matrix(self, y, n_samples):
        """Build ``(F, b, classes, mapping)`` from ``y`` (1-D labels or 2-D label memberships)."""
        c = self.n_clusters
        if y is None:
            F = np.zeros((n_samples, c))
            return F, np.zeros(n_samples, bool), np.empty(0, dtype=int), np.full(c, -1)
        y = np.asarray(y)
        if y.ndim == 2:
            F = check_array(y, dtype=float, ensure_min_samples=0)
            if F.shape != (n_samples, c):
                raise ValueError(f"A label membership matrix must have shape ({n_samples}, {c}).")
            if np.any(F < 0):
                raise ValueError("Label memberships must be non-negative.")
            s = F.sum(axis=1)
            b = s > 0
            F = F.copy()
            F[b] /= s[b, None]
            return F, b, np.arange(c), np.arange(c)
        y = np.asarray(y).ravel()
        if y.shape[0] != n_samples:
            raise ValueError(f"y has {y.shape[0]} entries, expected {n_samples}.")
        b = y != -1
        classes = np.unique(y[b])
        if classes.size > c:
            raise ValueError(
                f"y has {classes.size} classes but n_clusters={c}; every class needs a cluster."
            )
        F = np.zeros((n_samples, c))
        for j, cls in enumerate(classes):
            F[(y == cls), j] = 1.0
        mapping = np.full(c, -1)
        mapping[: classes.size] = classes
        return F, b, classes, mapping

    def fit(self, X, y=None):
        """Cluster ``X`` steered by the partial labels ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) or (n_samples, n_clusters), optional
            Class labels with ``-1`` for unlabeled samples, or a matrix of
            label memberships :math:`f_{ik}` whose zero rows mark the
            unlabeled samples. ``None`` means no supervision (plain FCM).

        Returns
        -------
        self : object
        """
        X_arr = validate_data(self, X, dtype=float, ensure_min_samples=2)
        F, b, classes, mapping = self._label_matrix(y, X_arr.shape[0])
        self.classes_ = classes
        self.label_mapping_ = mapping
        self.label_membership_ = F
        self.labelled_mask_ = b
        M = int(b.sum())
        if isinstance(self.alpha, str) and self.alpha == "auto":
            self.alpha_ = float(X_arr.shape[0] / M) if M > 0 else 0.0
        else:
            self.alpha_ = float(self.alpha) if not isinstance(self.alpha, str) else 0.0
        self._F_active = F
        self._b_active = b.astype(float)
        return super().fit(X)  # the original input keeps the feature names

    # ------------------------------------------------------------------ AO hooks
    def _init_state(self, X, rng):
        if isinstance(self.init, str) and self.init == "labels":
            c, n = self.n_clusters, X.shape[1]
            F, b = self._F_active, self._b_active > 0
            mean = X.mean(axis=0)
            half_range = np.abs(X - mean).max(axis=0)
            V = 2.0 * half_range * (rng.rand(c, n) - 0.5) + mean
            for i in range(c):
                w = F[b, i]
                if w.sum() > 0:
                    V[i] = (w[:, None] * X[b]).sum(axis=0) / w.sum()
            return self._init_memberships(sq_euclidean(X, V))
        return super()._init_state(X, rng)

    def _supervised(self, d2):
        return getattr(self, "_F_active", None) is not None and d2.shape[0] == self._F_active.shape[0]

    def _update_prototypes(self, X, Um):
        U = np.sqrt(Um)
        if self._supervised(Um):
            W = Um + self.alpha_ * (U - self._F_active * self._b_active[:, None]) ** 2
        else:  # pragma: no cover - fit always runs supervised
            W = Um
        s = W.sum(axis=0)
        s = np.where(s == 0, EPS, s)
        self.cluster_centers_ = (W.T @ X) / s[:, None]

    def _distances(self, X):
        return sq_euclidean(X, self.cluster_centers_)

    def _memberships(self, d2):
        if not self._supervised(d2):
            return memberships_from_distances(d2, 2.0)
        alpha = self.alpha_
        F, b = self._F_active, self._b_active
        inv = (d2 + EPS) ** -1.0
        fcm = inv / inv.sum(axis=1, keepdims=True)  # 1 / sum_j (D_ik^2 / D_jk^2)
        free = 1.0 + alpha * (1.0 - b * F.sum(axis=1))  # 1 + alpha(1 - b_k sum_j f_jk)
        U = (free[:, None] * fcm + alpha * F * b[:, None]) / (1.0 + alpha)
        return U

    def _objective(self, Um, d2):
        J = float(np.sum(Um * d2))
        if self._supervised(Um):
            U = np.sqrt(Um)
            J += self.alpha_ * float(np.sum((U - self._F_active * self._b_active[:, None]) ** 2 * d2))
        return J

    def _finalize(self, X, Um):
        self.covariances_ = fuzzy_covariances(X, Um, self.cluster_centers_)
        self._F_active = None
        self._b_active = None

    # ------------------------------------------------------------------ inference
    def predict_classes(self, X):
        """Class label of (new) samples: the class tied to the most likely cluster.

        Returns ``-1`` where the winning cluster is unsupervised.
        """
        check_is_fitted(self, "cluster_centers_")
        return self.label_mapping_[self.predict(X)]

    def score_labels(self, X, y):
        """Fraction of the labeled samples of ``y`` (``!= -1``) whose class is predicted."""
        y = np.asarray(y).ravel()
        mask = y != -1
        if not mask.any():
            return float("nan")
        pred = self.predict_classes(X)
        return float(np.mean(pred[mask] == y[mask]))
