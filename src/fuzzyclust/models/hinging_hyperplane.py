# SPDX-License-Identifier: MIT
# Copyright (c) 2026 János Abonyi
r"""Hinging-hyperplane regression tree identified by fuzzy c-regression clustering."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data

from .._utils import EPS, memberships_from_distances
from .fuzzy_c_regression import FuzzyCRegression
from .modified_gg import _extend, _require_y, weighted_least_squares

_GROWTH = ("balanced", "greedy")
_LEAF_MODELS = ("hinge", "linear")


@dataclass
class HingeNode:
    """One node of a :class:`HingingHyperplaneTree`.

    Attributes
    ----------
    theta : ndarray of shape (2, n_features + 1) or (1, n_features + 1)
        The two hyperplanes :math:`\\boldsymbol\\theta_1,\\boldsymbol\\theta_2`
        of the hinge (a single plane for a linear leaf).
    kind : {"max", "min", "linear"}
        Hinge function :math:`\\max` / :math:`\\min` of the two planes, or a
        plain linear model.
    depth, n_samples, sse, rmse, condition : node statistics
        ``condition`` is the hinge-identification quality :math:`\\varrho`
        (1 = the hinge split agrees perfectly with the fuzzy memberships).
    left, right : HingeNode or None
        Children; ``left`` receives the samples where plane 1 is active.
    """

    theta: np.ndarray
    kind: str
    depth: int
    n_samples: int
    sse: float
    rmse: float
    condition: float
    node_id: int = 0
    left: HingeNode | None = field(default=None, repr=False)
    right: HingeNode | None = field(default=None, repr=False)

    @property
    def is_leaf(self) -> bool:
        return self.left is None

    def hinge_value(self, Xe: np.ndarray) -> np.ndarray:
        """Value of the node model for extended regressors ``Xe``."""
        if self.kind == "linear":
            return Xe @ self.theta[0]
        p = Xe @ self.theta.T
        return p.max(axis=1) if self.kind == "max" else p.min(axis=1)

    def plane1_active(self, Xe: np.ndarray) -> np.ndarray:
        """Boolean mask of the samples on the side of the hinge where plane 1 is active."""
        if self.kind == "linear":
            return np.ones(Xe.shape[0], dtype=bool)
        h = Xe @ (self.theta[0] - self.theta[1])
        return h >= 0 if self.kind == "max" else h <= 0


class HingingHyperplaneTree(RegressorMixin, BaseEstimator):
    r"""Binary regression tree with hinging-hyperplane nodes identified by fuzzy c-regression.

    A hinging hyperplane (Breiman 1993) is the continuous piecewise-linear
    function :math:`h(\mathbf{x}) = \max(\mathbf{x}_e^T\boldsymbol\theta_1,\
    \mathbf{x}_e^T\boldsymbol\theta_2)`
    (or :math:`\min`) of two planes glued together along their intersection,
    the *hinge* :math:`\{\mathbf{x}: \mathbf{x}_e^T(\boldsymbol\theta_1-\boldsymbol\theta_2)=0\}`.
    Kenesei and Abonyi [1]_ observed that fuzzy c-regression with :math:`c=2`
    identifies exactly two local linear models and their fuzzy operating
    regions, hence one hinge; the sign of the hinge function then splits the
    data into the two operating regions, each of which is modeled again by a
    hinge, and so on. The result is a binary tree whose internal nodes split
    along oblique hinges (no choice of a splitting variable is needed) and
    whose terminal nodes are hinge functions, i.e. a piecewise-linear model
    with :math:`2\times` (number of leaves) affine pieces that is easy to use
    in model-based control.

    Parameters
    ----------
    max_depth : int, default=3
        Depth of the tree (the root has depth 0).
    min_samples_leaf : int, default=10
        A node is split only if both children receive at least this many samples.
    growth : {"balanced", "greedy"}, default="balanced"
        ``"balanced"`` grows every branch to ``max_depth``; ``"greedy"``
        repeatedly splits the leaf with the largest sum of squared errors
        until ``max_leaves`` leaves exist (or no leaf can be split).
    max_leaves : int, optional
        Number of leaves of the greedy tree (default :math:`2^{\text{max\_depth}}`).
    tol_rmse : float, default=0.0
        A node whose RMSE is below this value is not split further.
    leaf_model : {"hinge", "linear"}, default="hinge"
        Model stored in the terminal nodes. ``"hinge"`` keeps the hinge
        function identified at the leaf (two affine pieces per leaf, as in
        [1]_); ``"linear"`` replaces it by one least-squares plane fitted to
        the leaf's data, which gives a classical piecewise-linear regression
        tree with oblique splits (the tree is grown with hinges in either
        case, the replacement happens after growing).
    constrain_hinge : bool, default=True
        Enforce the crossing constraint of [1]_ (eq. 11): the membership
        weighted centers of the two clusters must lie on opposite sides of
        the hinge, so that the hinge lies inside the operating region. When
        the unconstrained FCRM solution violates it, the local models are
        re-estimated by constrained (quadratic programming) weighted least
        squares in a few alternating-optimization steps.
    m : float, default=2.0
        Fuzziness exponent of the fuzzy c-regression at every node.
    n_init : int, default=3
        Number of random FCRM restarts per node (the best objective is kept).
    fcrm_max_iter : int, default=100
    fcrm_tol : float, default=1e-4
    random_state : int, RandomState instance or None, default=None

    Attributes
    ----------
    tree_ : :class:`HingeNode`
        Root of the tree.
    depth_ : int
        Depth actually reached.
    n_leaves_, n_nodes_ : int
    leaves_ : list of :class:`HingeNode`

    Notes
    -----
    At a node with data :math:`\{(\mathbf{x}_k, y_k)\}` the fuzzy c-regression
    functional :math:`\sum_{i=1}^{2}\sum_k \mu_{ik}^m (y_k-\mathbf{x}_{e,k}^T\boldsymbol\theta_i)^2`
    is minimized (weighted least squares for :math:`\boldsymbol\theta_i`,
    FCM update for :math:`\mu_{ik}`). With :math:`\mathbf{v}_i` the
    membership-weighted mean of the regressors of cluster :math:`i`, the
    hinge constraint :math:`\mathbf{v}_{e,1}^T(\boldsymbol\theta_1-\boldsymbol\theta_2)` and
    :math:`\mathbf{v}_{e,2}^T(\boldsymbol\theta_1-\boldsymbol\theta_2)` of opposite
    signs guarantees that the two planes cross inside the data. The hinge
    type (:math:`\max` or :math:`\min`) is the one with the smaller squared
    error on the node data; the samples where plane 1 is active go to the
    left child. The quality of the identification at a node is

    .. math::
        \varrho = 1 - \frac{|m_1 - \sum_k \mu_{1k}|}{\sum_{i}\sum_k \mu_{ik}},

    with :math:`m_1` the number of samples on the side of plane 1; a value
    close to one means that the crisp hinge split agrees with the fuzzy
    partition. Growing stops at ``max_depth``, when a child would be too
    small, when the error does not decrease, or when the node RMSE is below
    ``tol_rmse``.

    References
    ----------
    .. [1] T. Kenesei, J. Abonyi, "Hinging hyperplane based regression tree
           identified by fuzzy clustering and its application", Applied Soft
           Computing, 13(2), 782–792, 2013.
    .. [2] L. Breiman, "Hinging hyperplanes for regression, classification and
           function approximation", IEEE Trans. Information Theory, 39(3),
           999–1013, 1993.
    .. [3] R. J. Hathaway, J. C. Bezdek, "Switching regression models and fuzzy
           clustering", IEEE Trans. Fuzzy Systems, 1(3), 195–204, 1993.

    Examples
    --------
    >>> from fuzzyclust.models import HingingHyperplaneTree, make_piecewise_linear
    >>> X, y = make_piecewise_linear(n_samples=300, random_state=0)
    >>> tree = HingingHyperplaneTree(max_depth=2, random_state=0).fit(X, y)
    >>> tree.n_leaves_ <= 4 and tree.score(X, y) > 0.95
    True
    """

    def __init__(
        self,
        max_depth: int = 3,
        min_samples_leaf: int = 10,
        growth: str = "balanced",
        max_leaves: int | None = None,
        tol_rmse: float = 0.0,
        leaf_model: str = "hinge",
        constrain_hinge: bool = True,
        m: float = 2.0,
        n_init: int = 3,
        fcrm_max_iter: int = 100,
        fcrm_tol: float = 1e-4,
        random_state=None,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.growth = growth
        self.max_leaves = max_leaves
        self.tol_rmse = tol_rmse
        self.leaf_model = leaf_model
        self.constrain_hinge = constrain_hinge
        self.m = m
        self.n_init = n_init
        self.fcrm_max_iter = fcrm_max_iter
        self.fcrm_tol = fcrm_tol
        self.random_state = random_state

    # ------------------------------------------------------------------ node identification
    def _constrained_wls(self, Xe, y, Um, theta0, v1, v2, sign):
        """Weighted LS of two planes subject to v1.(t1-t2)*sign <= -delta, v2.(t1-t2)*sign >= delta."""
        n1 = Xe.shape[1]
        W = Um
        H_blocks = [Xe.T @ (Xe * W[:, i][:, None]) for i in range(2)]
        c_blocks = [Xe.T @ (W[:, i] * y) for i in range(2)]
        scale = max(np.abs(y).max(), 1.0)
        delta = 1e-4 * scale

        def fun(t):
            t1, t2 = t[:n1], t[n1:]
            return 0.5 * (t1 @ H_blocks[0] @ t1 + t2 @ H_blocks[1] @ t2) - c_blocks[0] @ t1 - c_blocks[1] @ t2

        def jac(t):
            t1, t2 = t[:n1], t[n1:]
            return np.concatenate([H_blocks[0] @ t1 - c_blocks[0], H_blocks[1] @ t2 - c_blocks[1]])

        d = np.concatenate([v1, -v1])  # d.t = v1.(t1 - t2)
        e = np.concatenate([v2, -v2])
        cons = [
            {"type": "ineq", "fun": lambda t: -sign * (d @ t) - delta, "jac": lambda t: -sign * d},
            {"type": "ineq", "fun": lambda t: sign * (e @ t) - delta, "jac": lambda t: sign * e},
        ]
        res = minimize(
            fun, theta0.ravel(), jac=jac, constraints=cons, method="SLSQP", options={"maxiter": 200}
        )
        return res.x.reshape(2, n1), res.fun

    def _fit_hinge(self, X, y, rng):
        """Identify one hinge (two crossing planes) on the node data."""
        N, n = X.shape
        Xe = _extend(X)
        if N < 2 * (n + 1) + 2:
            theta = weighted_least_squares(Xe, y[:, None], np.ones(N))[:, 0]
            return theta[None, :], "linear", 1.0, None
        best = None
        for _ in range(max(1, self.n_init)):
            fcrm = FuzzyCRegression(
                n_clusters=2, m=self.m, tol=self.fcrm_tol, max_iter=self.fcrm_max_iter, random_state=rng
            ).fit(X, y)
            if best is None or fcrm.objective_ < best.objective_:
                best = fcrm
        theta = best.local_models_.copy()
        U = best.membership_.copy()
        if self.constrain_hinge:
            for _ in range(20):
                Um = U**self.m
                s = np.maximum(Um.sum(axis=0), EPS)
                v1 = (Um[:, 0] @ Xe) / s[0]
                v2 = (Um[:, 1] @ Xe) / s[1]
                delta = theta[0] - theta[1]
                h1, h2 = v1 @ delta, v2 @ delta
                if h1 * h2 < 0:
                    break
                cands = [self._constrained_wls(Xe, y, Um, theta, v1, v2, sgn) for sgn in (1.0, -1.0)]
                theta = min(cands, key=lambda c: c[1])[0]
                r2 = (y[:, None] - Xe @ theta.T) ** 2
                U = memberships_from_distances(r2, self.m)
        # hinge type: max or min, whichever fits the node data better
        P = Xe @ theta.T
        sse_max = float(np.sum((y - P.max(axis=1)) ** 2))
        sse_min = float(np.sum((y - P.min(axis=1)) ** 2))
        kind = "max" if sse_max <= sse_min else "min"
        # identification quality rho (eq. 22 of the paper)
        h = Xe @ (theta[0] - theta[1])
        active1 = h >= 0 if kind == "max" else h <= 0
        rho = 1.0 - abs(active1.sum() - U[:, 0].sum()) / max(U.sum(), EPS)
        return theta, kind, float(rho), U

    def _make_node(self, X, y, depth, rng):
        theta, kind, rho, _ = self._fit_hinge(X, y, rng)
        node = HingeNode(
            theta=theta, kind=kind, depth=depth, n_samples=X.shape[0], sse=0.0, rmse=0.0, condition=rho
        )
        Xe = _extend(X)
        res = y - node.hinge_value(Xe)
        node.sse = float(res @ res)
        node.rmse = float(np.sqrt(node.sse / max(X.shape[0], 1)))
        return node

    def _can_split(self, node, X, y):
        if node.kind == "linear" or node.depth >= self.max_depth or node.rmse <= self.tol_rmse:
            return None
        mask = node.plane1_active(_extend(X))
        nl, nr = int(mask.sum()), int((~mask).sum())
        if min(nl, nr) < max(self.min_samples_leaf, 1):
            return None
        return mask

    def _try_split(self, node, X, y, rng):
        """Fit the two children; return them if the error decreases, else None."""
        mask = self._can_split(node, X, y)
        if mask is None:
            return None
        left = self._make_node(X[mask], y[mask], node.depth + 1, rng)
        right = self._make_node(X[~mask], y[~mask], node.depth + 1, rng)
        if left.sse + right.sse >= node.sse * (1.0 - 1e-9):
            return None
        return left, right, mask

    def _grow_balanced(self, node, X, y, rng):
        split = self._try_split(node, X, y, rng)
        if split is None:
            return
        left, right, mask = split
        node.left, node.right = left, right
        self._grow_balanced(left, X[mask], y[mask], rng)
        self._grow_balanced(right, X[~mask], y[~mask], rng)

    def _grow_greedy(self, root, X, y, rng):
        max_leaves = self.max_leaves if self.max_leaves is not None else 2**self.max_depth
        leaves = [(root, X, y)]
        while len(leaves) < max_leaves:
            order = np.argsort([-leaf[0].sse for leaf in leaves])
            done = False
            for idx in order:
                node, Xn, yn = leaves[idx]
                if getattr(node, "_unsplittable", False):
                    continue
                split = self._try_split(node, Xn, yn, rng)
                if split is None:
                    node._unsplittable = True
                    continue
                left, right, mask = split
                node.left, node.right = left, right
                leaves.pop(idx)
                leaves += [(left, Xn[mask], yn[mask]), (right, Xn[~mask], yn[~mask])]
                done = True
                break
            if not done:
                break
        for leaf in leaves:
            if hasattr(leaf[0], "_unsplittable"):
                del leaf[0]._unsplittable

    # ------------------------------------------------------------------ public API
    def fit(self, X, y=None):
        """Grow the tree on the regression data ``(X, y)``."""
        _require_y("HingingHyperplaneTree", y)
        X, y = validate_data(self, X, y, dtype=float, y_numeric=True, ensure_min_samples=2)
        if self.growth not in _GROWTH:
            raise ValueError(f"growth must be one of {_GROWTH}.")
        if self.leaf_model not in _LEAF_MODELS:
            raise ValueError(f"leaf_model must be one of {_LEAF_MODELS}.")
        if self.max_depth < 0:
            raise ValueError("max_depth must be non-negative.")
        if self.m <= 1.0:
            raise ValueError("m must be > 1.")
        rng = check_random_state(self.random_state)
        root = self._make_node(X, y, 0, rng)
        if self.growth == "balanced":
            self._grow_balanced(root, X, y, rng)
        else:
            self._grow_greedy(root, X, y, rng)
        self.tree_ = root
        nodes = self._nodes()
        for i, node in enumerate(nodes):
            node.node_id = i
        self.n_nodes_ = len(nodes)
        self.leaves_ = [nd for nd in nodes if nd.is_leaf]
        self.n_leaves_ = len(self.leaves_)
        self.depth_ = max(nd.depth for nd in nodes)
        if self.leaf_model == "linear":
            self._linearize_leaves(X, y)
        return self

    def _linearize_leaves(self, X, y):
        """Replace the hinge of every leaf by one least-squares plane fitted to the leaf's data."""
        Xe = _extend(X)
        leaf_of = self.apply(X)
        for i, leaf in enumerate(self.leaves_):
            mask = leaf_of == i
            if not mask.any():
                continue
            theta = weighted_least_squares(Xe[mask], y[mask, None], np.ones(int(mask.sum())))[:, 0]
            leaf.theta = theta[None, :]
            leaf.kind = "linear"
            res = y[mask] - Xe[mask] @ theta
            leaf.sse = float(res @ res)
            leaf.rmse = float(np.sqrt(leaf.sse / mask.sum()))

    def _nodes(self):
        out, stack = [], [self.tree_]
        while stack:
            node = stack.pop(0)
            out.append(node)
            if not node.is_leaf:
                stack += [node.left, node.right]
        return out

    def _predict_node(self, node, Xe, idx, out):
        if node.is_leaf:
            out[idx] = node.hinge_value(Xe)
            return
        mask = node.plane1_active(Xe)
        if mask.any():
            self._predict_node(node.left, Xe[mask], idx[mask], out)
        if (~mask).any():
            self._predict_node(node.right, Xe[~mask], idx[~mask], out)

    def predict(self, X):
        """Piecewise-linear prediction, shape ``(n_samples,)``."""
        check_is_fitted(self, "tree_")
        X = validate_data(self, X, dtype=float, reset=False)
        Xe = _extend(X)
        out = np.empty(X.shape[0])
        self._predict_node(self.tree_, Xe, np.arange(X.shape[0]), out)
        return out

    def apply(self, X):
        """Index (in ``leaves_``) of the leaf every sample falls into."""
        check_is_fitted(self, "tree_")
        X = validate_data(self, X, dtype=float, reset=False)
        Xe = _extend(X)
        out = np.empty(X.shape[0], dtype=int)
        leaf_ids = {id(leaf): i for i, leaf in enumerate(self.leaves_)}

        def rec(node, Xe_, idx):
            if node.is_leaf:
                out[idx] = leaf_ids[id(node)]
                return
            mask = node.plane1_active(Xe_)
            if mask.any():
                rec(node.left, Xe_[mask], idx[mask])
            if (~mask).any():
                rec(node.right, Xe_[~mask], idx[~mask])

        rec(self.tree_, Xe, np.arange(X.shape[0]))
        return out

    def node_table(self) -> pd.DataFrame:
        """Statistics of every node (depth, samples, RMSE, hinge condition, leaf flag)."""
        check_is_fitted(self, "tree_")
        rows = [
            {
                "node": nd.node_id,
                "depth": nd.depth,
                "kind": nd.kind,
                "n_samples": nd.n_samples,
                "rmse": nd.rmse,
                "condition": nd.condition,
                "leaf": nd.is_leaf,
            }
            for nd in self._nodes()
        ]
        return pd.DataFrame(rows)

    def describe(self, feature_names=None, digits: int = 3) -> str:
        """Readable traversal of the tree (hinge planes, split rule and node statistics)."""
        check_is_fitted(self, "tree_")
        n = self.n_features_in_
        if feature_names is None:
            feature_names = list(getattr(self, "feature_names_in_", [f"x{j + 1}" for j in range(n)]))
        f = f"{{:+.{digits}g}}"

        def plane(th):
            return (
                " ".join(f"{f.format(th[j])}*{feature_names[j]}" for j in range(n)) + f" {f.format(th[-1])}"
            )

        lines = []

        def rec(node, indent):
            pad = "  " * indent
            head = (
                f"{pad}node {node.node_id} (depth {node.depth}, N={node.n_samples}, "
                f"RMSE={node.rmse:.{digits}g}"
            )
            if node.kind == "linear":
                lines.append(head + f"): y = {plane(node.theta[0])}")
                return
            lines.append(head + f", rho={node.condition:.2f}): y = {node.kind}(h1, h2)")
            lines.append(f"{pad}  h1 = {plane(node.theta[0])}")
            lines.append(f"{pad}  h2 = {plane(node.theta[1])}")
            if not node.is_leaf:
                lines.append(f"{pad}  if h1 {'>=' if node.kind == 'max' else '<='} h2 (plane 1 active):")
                rec(node.left, indent + 2)
                lines.append(f"{pad}  else:")
                rec(node.right, indent + 2)

        rec(self.tree_, 0)
        return "\n".join(lines)


__all__ = ["HingingHyperplaneTree", "HingeNode"]
