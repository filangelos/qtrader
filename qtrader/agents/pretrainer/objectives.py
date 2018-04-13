import numpy as np


def _mu_p(w: np.ndarray, r: np.ndarray) -> float:
    """Portfolio Returns."""
    return np.dot(w.T, r)


def _sigma_p(w: np.ndarray, Sigma: np.ndarray) -> float:
    """Portoflio Variance"""
    return np.dot(np.dot(w.T, Sigma), w)


def _trans_costs(w: np.ndarray, w0: np.ndarray, coef: float) -> float:
    """Transaction Costs."""
    return np.sum(np.abs(w0 - w)) * coef


def risk_aversion(w: np.ndarray, mu: np.ndarray,
                  Sigma: np.ndarray, w0: np.ndarray,
                  alpha: float, beta: float) -> float:
    """Risk Aversion with Transaction Costs."""
    assert Sigma.shape[0] == Sigma.shape[1]
    assert mu.shape[0] == Sigma.shape[0]
    assert w.shape == w0.shape
    # mean - alpha * variance - transaction_costs
    return - (_mu_p(w, mu) - alpha * _sigma_p(w, Sigma) - _trans_costs(w, w0, beta))


def sharpe_ratio(w: np.ndarray, mu: np.ndarray,
                 Sigma: np.ndarray, w0: np.ndarray,
                 gamma: float) -> float:
    """Sharpe Ratio with Transaction Costs."""
    assert Sigma.shape[0] == Sigma.shape[1]
    assert mu.shape[0] == Sigma.shape[0]
    assert w.shape == w0.shape
    # mean - alpha * variance - transaction_costs
    return - (_mu_p(w, mu) / _sigma_p(w, Sigma) - _trans_costs(w, w0, gamma))
