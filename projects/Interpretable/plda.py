# -*- coding: utf-8 -*-
"""
Interpretable / plda.py
=======================
自实现的 **two-covariance Gaussian PLDA**（Brümmer & de Villiers 2010）+
参考代码里的 **PCA-PLDA-EER 分类器**。

为什么自实现：参考代码 `PCA_PLDA_EER_Classifier.py` 依赖
`speechbrain.processing.PLDA_LDA`，而 speechbrain 会拖入 `torchaudio`
（仓库明令禁止）。这里用纯 numpy 复刻同一套模型与打分，避免新依赖。

模型：对类别 i 的隐变量 y_i ~ N(mu, B)，观测 x_ij | y_i ~ N(y_i, W)；
等价的因子形式 x = mu + F h + eps，h~N(0,I)，eps~N(0,Sigma)，
其中 B = F Fᵀ（类间），Sigma = W（类内）。用 EM 估 F、Sigma。

打分（单条注册 vs 单条测试）：
    LLR(x1,x2) = log N([x1;x2] | [mu;mu], [[B+S, B],[B, B+S]])
               - log N(x1|mu, B+S) - log N(x2|mu, B+S)
与 speechbrain 的 `fast_PLDA_scoring` 同源。
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler


# ===========================================================================
# two-covariance Gaussian PLDA
# ===========================================================================
def _logdet_and_quad(m: np.ndarray, cov: np.ndarray) -> float:
    """log N(m | 0, cov)，带极小的 jitter 保证正定。"""
    d = cov.shape[0]
    jitter = 1e-8 * (np.trace(cov) / d + 1e-12)
    cov = cov + jitter * np.eye(d)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        cov = cov + (1e-4 + abs(logdet)) * np.eye(d)
        sign, logdet = np.linalg.slogdet(cov)
    sol = np.linalg.solve(cov, m)
    return -0.5 * (d * np.log(2 * np.pi) + logdet + float(m @ sol))


class TwoCovariancePLDA(BaseEstimator):
    """two-covariance PLDA（因子形式，EM 估计 F / Sigma / mu）。

    参数
    ----
    rank_f : int | None
        因子维度。None → 取特征维度（与参考代码 `PLDA(rank_f=dim)` 一致）。
    n_iter : int
        EM 迭代次数。
    seed : int
    """

    def __init__(self, rank_f: int | None = None, n_iter: int = 40, seed: int = 0):
        self.rank_f = rank_f
        self.n_iter = n_iter
        self.seed = seed

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, labels: np.ndarray) -> "TwoCovariancePLDA":
        X = np.asarray(X, dtype=np.float64)
        labels = np.asarray(labels)
        rng = np.random.default_rng(self.seed)
        N, D = X.shape
        self.mu_ = X.mean(axis=0)
        Z = X - self.mu_

        cls = np.unique(labels)
        idx = [np.where(labels == c)[0] for c in cls]
        r = self.rank_f or D
        r = int(min(r, D, max(N - len(cls), 1)))

        # ---- 初始化：类内协方差 -> Sigma；类间协方差 -> F Fᵀ ----
        Sigma = np.zeros((D, D))
        for ix in idx:
            Zi = Z[ix] - Z[ix].mean(axis=0, keepdims=True)
            Sigma += Zi.T @ Zi
        Sigma /= max(N - len(cls), 1)
        Sigma += 1e-3 * np.eye(D)

        means = np.vstack([Z[ix].mean(axis=0) for ix in idx])          # [C, D]
        B = np.cov(means, rowvar=False) if len(means) > 1 else np.eye(D)
        B = np.atleast_2d(B) + 1e-3 * np.eye(D)
        # F 使 F Fᵀ = B
        w, V = np.linalg.eigh(B)
        w = np.clip(w, 0.0, None)
        order = np.argsort(w)[::-1][:r]
        F = V[:, order] * np.sqrt(w[order])[None, :]                   # [D, r]

        # ---- EM ----
        for _ in range(self.n_iter):
            Sig_inv = np.linalg.inv(Sigma)
            FtSiF = F.T @ Sig_inv @ F                                  # [r, r]
            A = np.zeros((D, r))
            Bmat = np.zeros((r, r))
            ZZt = np.zeros((D, D))
            for ix in idx:
                n_i = len(ix)
                Zi = Z[ix]
                S = Zi.sum(axis=0)                                     # [D]
                Cov_h = np.linalg.inv(np.eye(r) + n_i * FtSiF)
                Eh = Cov_h @ (F.T @ (Sig_inv @ S))                     # [r]
                Ehh = Cov_h + np.outer(Eh, Eh)
                A += np.outer(S, Eh)
                Bmat += n_i * Ehh
                ZZt += Zi.T @ Zi
            F_new = A @ np.linalg.pinv(Bmat)

            # Sigma = 1/N * sum_ij E[(z - F h)(z - F h)ᵀ]
            # = 1/N [ ZZᵀ - A Fᵀ - F Aᵀ + F Bmat Fᵀ ]
            Sigma_new = (ZZt - A @ F_new.T - F_new @ A.T + F_new @ Bmat @ F_new.T) / N
            Sigma_new = 0.5 * (Sigma_new + Sigma_new.T)
            Sigma_new += 1e-6 * np.eye(D)
            F = F_new
            Sigma = Sigma_new

        self.F_ = np.asarray(F, dtype=np.float64)
        self.Sigma_ = np.asarray(Sigma, dtype=np.float64)
        self.B_ = self.F_ @ self.F_.T
        self.Sigma_ac_ = self.B_ + self.Sigma_
        return self

    # ------------------------------------------------------------------
    def llr(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """单条 x1（测试）与单条 x2（注册）之间的 PLDA 对数似然比。"""
        x1 = np.asarray(x1, dtype=np.float64).ravel() - self.mu_
        x2 = np.asarray(x2, dtype=np.float64).ravel() - self.mu_
        # 同源：联合 ~ N(0, [[Sac, B],[B, Sac]])
        joint = np.block([[self.Sigma_ac_, self.B_], [self.B_, self.Sigma_ac_]])
        m_j = np.concatenate([x1, x2])
        ll_same = _logdet_and_quad(m_j, joint)
        # 异源：两个独立边缘
        ll_diff = _logdet_and_quad(x1, self.Sigma_ac_) + _logdet_and_quad(x2, self.Sigma_ac_)
        return float(ll_same - ll_diff)


# ===========================================================================
# PCA + PLDA + EER 分类器（复刻参考 PCA_PLDA_EER_Classifier）
# ===========================================================================
class PCAPLDAEERClassifier(BaseEstimator, ClassifierMixin):
    """参考代码 `PCA_PLDA_EER_Classifier` 的等价实现（无 speechbrain 依赖）。

    fit(X, y)：
      * （可选）StandardScaler → PCA(n_components=PCA_n)
      * PLDA 建模（two-covariance）
      * 两类注册向量 = 各类均值；用训练集分数求 EER 阈值
    predict_scores(X)：返回 [N] 的分数 = LLR(测试, 类1均值) − LLR(测试, 类0均值)
    predict(X)：分数 > EER 阈值 → 1，否则 0
    """

    def __init__(self, PCA_n: int = 20, normalize: int = 0, n_iter: int = 40, seed: int = 0):
        self.PCA_n = PCA_n
        self.normalize = normalize
        self.n_iter = n_iter
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PCAPLDAEERClassifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel().astype(int)
        self.classes_ = np.unique(y)

        if self.normalize:
            self.scaler_ = StandardScaler().fit(X)
            Xt = self.scaler_.transform(X)
        else:
            self.scaler_ = None
            Xt = X

        n_comp = int(min(self.PCA_n, Xt.shape[0] - 1, Xt.shape[1]))
        n_comp = max(n_comp, 1)
        self.pca_ = PCA(n_components=n_comp, random_state=self.seed).fit(Xt)
        Z = self.pca_.transform(Xt)

        self.plda_ = TwoCovariancePLDA(rank_f=n_comp, n_iter=self.n_iter,
                                       seed=self.seed).fit(Z, y)
        # 两类注册（enrollment）：类别均值
        self.en0_ = Z[y == 0].mean(axis=0)
        self.en1_ = Z[y == 1].mean(axis=0)

        train_scores = self._scores(Z)
        self.eer_threshold_ = self._eer_threshold(y, train_scores)
        return self

    # ------------------------------------------------------------------
    def _scores(self, Z: np.ndarray) -> np.ndarray:
        return np.array([self.plda_.llr(z, self.en1_) - self.plda_.llr(z, self.en0_)
                         for z in Z], dtype=np.float64)

    @staticmethod
    def _eer_threshold(y: np.ndarray, scores: np.ndarray) -> float:
        fpr, tpr, thr = roc_curve(y, scores, pos_label=1, drop_intermediate=False)
        fnr = 1 - tpr
        i = int(np.nanargmin(np.abs(fnr - fpr)))
        # roc_curve 的阈值第一个是 +inf，取有限值
        return float(thr[i]) if np.isfinite(thr[i]) else float(np.nanmax(scores))

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        Xt = self.scaler_.transform(X) if self.scaler_ is not None else X
        Z = self.pca_.transform(Xt)
        return self._scores(Z)

    def predict(self, X: np.ndarray) -> np.ndarray:
        s = self.predict_scores(X)
        return (s > self.eer_threshold_).astype(int)
