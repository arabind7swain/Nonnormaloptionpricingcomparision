"""Helper routines for fitting and evaluating alternative distributions."""

from __future__ import annotations

import numpy as np
from scipy import stats
import time
import math
from typing import Optional, Callable
from scipy.optimize import minimize
from scipy.integrate import quad
from distributions import (
    champernowne_pdf,
    champernowne_tail,
    champernowne_base_stats,
    cgmy_pdf,
    CgmyParams,
    gsh_pdf,
    sgsh_pdf,
    nef_ghs_pdf,
    nef_ghs_tail_prob,
    vg_pdf,
)
from functools import lru_cache

SCIPY_ALIASES: dict[str, str | None] = {
    "normal": "norm",
    "hypsecant": "hypsecant",
    "logistic": "logistic",
    "laplace": "laplace",
    "cauchy": "cauchy",
    "asymmetric_laplace": "laplace_asymmetric",
    "skew_normal": "skewnorm",
    "hyperbolic": "genhyperbolic",
    "johnsonsu": "johnsonsu",
    "tukeylambda": "tukeylambda",
    "ged": "gennorm",
    "nig": "norminvgauss",
    "vg": "vargamma",
    "gh": "genhyperbolic",
    "levy_stable": "levy_stable",
    "genlogistic": "genlogistic",
    "powernorm": "powernorm",
    "crystalball": "crystalball",
    "skewt": "skewt",
}
FIT_CACHE: dict[str, dict[str, float | tuple]] = {}
CUSTOM_FITTERS: dict[str, Callable[[np.ndarray], tuple[dict[str, float | str], Callable[[np.ndarray], np.ndarray]]]] = {}
CHAMPER_LIMIT = 400
CGMY_QUAD_LIMIT = 200
CGMY_CDF_LOWER = -50.0
GSH_QUAD_LIMIT = 400
GSH_CDF_LOWER = -50.0
SGSH_QUAD_LIMIT = 400
SGSH_CDF_LOWER = -50.0
NEF_GHS_LIMIT = 200
VG_CDF_LOWER = -50.0
VG_QUAD_LIMIT = 200


def compute_density_moments(pdf_func: Callable[[np.ndarray], np.ndarray], data: np.ndarray) -> tuple[float, float, float, float]:
    """Return mean/std/skew/kurtosis by numerically integrating the fitted pdf."""
    std_est = float(np.std(data, ddof=1))
    span = 5.0 * std_est if std_est > 1e-6 else 5.0
    lo = float(np.min(data) - span)
    hi = float(np.max(data) + span)
    grid = np.linspace(lo, hi, 4000)
    pdf_vals = np.asarray(pdf_func(grid), dtype=float)
    pdf_vals = np.clip(pdf_vals, 0.0, np.inf)
    norm = np.trapezoid(pdf_vals, grid)
    if norm <= 0.0 or not np.isfinite(norm):
        return (float("nan"),) * 4
    pdf_vals = pdf_vals / norm
    mean = float(np.trapezoid(grid * pdf_vals, grid))
    var = float(np.trapezoid(((grid - mean) ** 2) * pdf_vals, grid))
    std = math.sqrt(var) if var > 0.0 else 0.0
    if std == 0.0 or not math.isfinite(std):
        return mean, 0.0, 0.0, 3.0
    centered = (grid - mean) / std
    skew = float(np.trapezoid((centered ** 3) * pdf_vals, grid))
    kurt = float(np.trapezoid((centered ** 4) * pdf_vals, grid))
    return mean, std, skew, kurt


@lru_cache(maxsize=None)
def _champer_norm_const(d: float) -> float:
    """Return normalization constant for Champernowne shape d."""
    norm_const, _ = champernowne_base_stats(d, CHAMPER_LIMIT)
    return norm_const


def simulate_mixture(
    n_samples: int,
    components: list[tuple[float, float, float]],
    seed: int | None = None,
) -> np.ndarray:
    """Return draws from a normal mixture with (weight, mean, std) components."""
    rng = np.random.default_rng(seed)
    weights = np.array([w for w, _, _ in components], dtype=float)
    weights = weights / weights.sum()
    picks = rng.choice(len(components), size=n_samples, p=weights)
    draws = np.empty(n_samples, dtype=float)
    for idx, (_, mean, std) in enumerate(components):
        mask = picks == idx
        draws[mask] = rng.normal(mean, std, mask.sum())
    return draws


def fit_distribution(
    data: np.ndarray, dist_name: str
) -> tuple[dict[str, float | str], Callable[[np.ndarray], np.ndarray]]:
    """Fit a distribution by name using SciPy or custom routines."""
    scipy_name = SCIPY_ALIASES.get(dist_name, dist_name)
    if scipy_name is None or not hasattr(stats, scipy_name):
        custom_fitter = CUSTOM_FITTERS.get(dist_name)
        if custom_fitter is None:
            raise NotImplementedError(f"{dist_name} does not have a SciPy analog yet")
        return custom_fitter(data)
    cached = FIT_CACHE.get(scipy_name)
    if cached is None:
        start = time.perf_counter()
        dist = getattr(stats, scipy_name)
        params = dist.fit(data)
        loglike = float(np.sum(dist.logpdf(data, *params)))
        ks_stat, ks_pvalue = stats.kstest(data, scipy_name, args=params)
        elapsed = time.perf_counter() - start
        moments = dist.stats(*params, moments="mvsk")
        moment_vals = [float(m) for m in moments]
        std_density = math.sqrt(moment_vals[1]) if moment_vals[1] > 0.0 else 0.0
        shapes_str = getattr(dist, "shapes", None)
        shape_names = [s.strip() for s in shapes_str.split(",")] if shapes_str else []
        param_names = shape_names + ["loc", "scale"]
        cached = {
            "loglike": loglike,
            "ks_stat": float(ks_stat),
            "ks_pvalue": float(ks_pvalue),
            "params": params,
            "n_params": len(params),
            "fit_time": elapsed,
            "mean_density": moment_vals[0],
            "std_density": std_density,
            "skew_density": moment_vals[2],
            "kurt_density": moment_vals[3],
            "param_names": param_names,
        }
        FIT_CACHE[scipy_name] = cached
    pdf_func = lambda grid, dist=scipy_name, params=cached["params"]: getattr(stats, dist).pdf(grid, *params)
    entry = {
        "distribution": dist_name,
        "scipy_name": scipy_name,
        **cached,
    }
    return entry, pdf_func


def fit_champernowne(data: np.ndarray) -> tuple[dict[str, float | str], Callable[[np.ndarray], np.ndarray]]:
    """Fit a Champernowne distribution via MLE."""
    start = time.perf_counter()
    mean0 = float(np.mean(data))
    std0 = float(np.std(data, ddof=1))
    init = np.array([mean0, np.log(max(std0, 1e-3)), 0.0])

    def nll(theta: np.ndarray) -> float:
        mu, log_scale, d = theta
        scale = math.exp(log_scale)
        if scale <= 0.0 or d <= -1.9:
            return np.inf
        norm_const = _champer_norm_const(float(d))
        z = (data - mu) / scale
        pdf_vals = np.array([champernowne_pdf(float(val), d, norm_const) for val in z]) / scale
        if np.any(pdf_vals <= 0.0):
            return np.inf
        return -float(np.sum(np.log(pdf_vals)))

    res = minimize(
        nll,
        init,
        method="L-BFGS-B",
        bounds=[(None, None), (None, None), (-1.9, 10.0)],
        options={"maxiter": 500},
    )
    if not res.success:
        raise RuntimeError(f"Champernowne fit failed: {res.message}")
    mu, log_scale, d = res.x
    scale = math.exp(log_scale)
    norm_const = _champer_norm_const(float(d))
    loglike = -res.fun

    def champer_cdf(x: float) -> float | np.ndarray:
        x_arr = np.atleast_1d(x).astype(float)
        vals = []
        for xi in x_arr:
            z = (xi - mu) / scale
            tail = champernowne_tail(z, d, norm_const, CHAMPER_LIMIT)
            vals.append(max(0.0, min(1.0, 1.0 - tail)))
        vals = np.array(vals)
        if np.ndim(x) == 0:
            return float(vals[0])
        return vals

    ks_stat, ks_pvalue = stats.kstest(data, champer_cdf)
    fit_time = time.perf_counter() - start

    def pdf_func(grid: np.ndarray) -> np.ndarray:
        z = (grid - mu) / scale
        vals = np.array([champernowne_pdf(float(val), d, norm_const) for val in np.atleast_1d(z)])
        vals = vals / scale
        if np.ndim(grid) == 0:
            return np.array([vals[0]])
        return vals

    mean_d, std_d, skew_d, kurt_d = compute_density_moments(pdf_func, data)
    record = {
        "distribution": "champernowne",
        "scipy_name": None,
        "loglike": float(loglike),
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "params": (float(mu), float(scale), float(d)),
        "n_params": 3,
        "fit_time": fit_time,
        "mean_density": mean_d,
        "std_density": std_d,
        "skew_density": skew_d,
        "kurt_density": kurt_d,
        "param_names": ["mu", "scale", "d"],
    }
    return record, pdf_func


CUSTOM_FITTERS["champernowne"] = fit_champernowne


def fit_cgmy(data: np.ndarray) -> tuple[dict[str, float | str], Callable[[np.ndarray], np.ndarray]]:
    """Fit a CGMY distribution via MLE."""
    start = time.perf_counter()
    mean0 = float(np.mean(data))
    std0 = float(np.std(data, ddof=1))
    theta0 = np.array([math.log(1.0), math.log(2.0), math.log(2.0), 0.5, mean0, math.log(max(std0 * 0.5, 1e-3))])

    def build_params(theta: np.ndarray) -> Optional[CgmyParams]:
        log_c, log_g, log_m, y_raw, mu_val, log_scale = theta
        C_val = math.exp(log_c)
        G_val = math.exp(log_g) + 1e-6
        M_val = math.exp(log_m) + 1e-6
        Y_val = 0.01 + 1.98 / (1.0 + math.exp(-y_raw))
        scale = math.exp(log_scale)
        if scale >= min(G_val, M_val):
            return None
        gamma_neg = math.gamma(-Y_val)
        if not math.isfinite(gamma_neg):
            return None
        return CgmyParams(C_val, G_val, M_val, Y_val, scale, mu_val, gamma_neg, 1.0)

    def nll(theta: np.ndarray) -> float:
        params = build_params(theta)
        if params is None:
            return np.inf
        pdf_vals = np.array([cgmy_pdf(float(xi), params, CGMY_QUAD_LIMIT) for xi in data])
        if np.any(pdf_vals <= 0.0) or np.any(~np.isfinite(pdf_vals)):
            return np.inf
        return -float(np.sum(np.log(pdf_vals)))

    res = minimize(
        nll,
        theta0,
        method="L-BFGS-B",
        options={"maxiter": 400},
    )
    if not res.success:
        raise RuntimeError(f"CGMY fit failed: {res.message}")
    params = build_params(res.x)
    if params is None:
        raise RuntimeError("CGMY fit ended with invalid parameters")
    loglike = -res.fun

    def cgmy_cdf_val(x: float | np.ndarray) -> float | np.ndarray:
        def single_value(xi: float) -> float:
            if xi <= CGMY_CDF_LOWER:
                return 0.0
            val, _ = quad(
                lambda y: cgmy_pdf(y, params, CGMY_QUAD_LIMIT),
                CGMY_CDF_LOWER,
                xi,
                limit=CGMY_QUAD_LIMIT,
                epsabs=1e-6,
                epsrel=1e-6,
            )
            return max(0.0, min(1.0, val))

        arr = np.atleast_1d(x).astype(float)
        vals = np.array([single_value(xi) for xi in arr])
        if np.ndim(x) == 0:
            return float(vals[0])
        return vals

    ks_stat, ks_pvalue = stats.kstest(data, cgmy_cdf_val)
    fit_time = time.perf_counter() - start

    def pdf_func(grid: np.ndarray) -> np.ndarray:
        grid_arr = np.atleast_1d(grid).astype(float)
        vals = np.array([cgmy_pdf(float(val), params, CGMY_QUAD_LIMIT) for val in grid_arr])
        if np.ndim(grid) == 0:
            return np.array([vals[0]])
        return vals

    mean_d, std_d, skew_d, kurt_d = compute_density_moments(pdf_func, data)
    mean_d, std_d, skew_d, kurt_d = compute_density_moments(pdf_func, data)
    record = {
        "distribution": "cgmy",
        "scipy_name": None,
        "loglike": float(loglike),
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "params": (params.C, params.G, params.M, params.Y, params.scale, params.mu),
        "n_params": 6,
        "fit_time": fit_time,
        "mean_density": mean_d,
        "std_density": std_d,
        "skew_density": skew_d,
        "kurt_density": kurt_d,
        "param_names": ["C", "G", "M", "Y", "scale", "mu"],
    }
    return record, pdf_func


CUSTOM_FITTERS["cgmy"] = fit_cgmy


def fit_gsh(data: np.ndarray) -> tuple[dict[str, float | str], Callable[[np.ndarray], np.ndarray]]:
    """Fit a generalized secant hyperbolic distribution."""
    start = time.perf_counter()
    mean0 = float(np.mean(data))
    std0 = max(float(np.std(data, ddof=1)), 1e-3)
    theta0 = np.array([mean0, math.log(std0), 0.0])

    def nll(theta: np.ndarray) -> float:
        mu, log_scale, t_param = theta
        scale = math.exp(log_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            return np.inf
        pdf_vals = np.array(
            [gsh_pdf(float((xi - mu) / scale), t_param, GSH_QUAD_LIMIT) for xi in data]
        ) / scale
        if np.any(pdf_vals <= 0.0) or np.any(~np.isfinite(pdf_vals)):
            return np.inf
        return -float(np.sum(np.log(pdf_vals)))

    res = minimize(
        nll,
        theta0,
        method="L-BFGS-B",
        bounds=[(None, None), (None, None), (-5.0, 5.0)],
        options={"maxiter": 400},
    )
    if not res.success:
        raise RuntimeError(f"GSH fit failed: {res.message}")
    mu, log_scale, t_param = res.x
    scale = math.exp(log_scale)
    loglike = -res.fun

    def gsh_cdf_val(x: float | np.ndarray) -> float | np.ndarray:
        def single_value(xi: float) -> float:
            if xi <= GSH_CDF_LOWER:
                return 0.0
            val, _ = quad(
                lambda y: gsh_pdf(float((y - mu) / scale), t_param, GSH_QUAD_LIMIT) / scale,
                GSH_CDF_LOWER,
                xi,
                limit=GSH_QUAD_LIMIT,
                epsabs=1e-6,
                epsrel=1e-6,
            )
            return max(0.0, min(1.0, val))

        arr = np.atleast_1d(x).astype(float)
        vals = np.array([single_value(xi) for xi in arr])
        if np.ndim(x) == 0:
            return float(vals[0])
        return vals

    ks_stat, ks_pvalue = stats.kstest(data, gsh_cdf_val)
    fit_time = time.perf_counter() - start

    def pdf_func(grid: np.ndarray) -> np.ndarray:
        z = np.atleast_1d((grid - mu) / scale).astype(float)
        vals = np.array([gsh_pdf(float(val), t_param, GSH_QUAD_LIMIT) for val in z]) / scale
        if np.ndim(grid) == 0:
            return np.array([vals[0]])
        return vals

    mean_d, std_d, skew_d, kurt_d = compute_density_moments(pdf_func, data)
    record = {
        "distribution": "gsh",
        "scipy_name": None,
        "loglike": float(loglike),
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "params": (float(mu), float(scale), float(t_param)),
        "n_params": 3,
        "fit_time": fit_time,
        "mean_density": mean_d,
        "std_density": std_d,
        "skew_density": skew_d,
        "kurt_density": kurt_d,
        "param_names": ["mu", "scale", "t"],
    }
    return record, pdf_func


CUSTOM_FITTERS["gsh"] = fit_gsh


def fit_sgsh(data: np.ndarray) -> tuple[dict[str, float | str], Callable[[np.ndarray], np.ndarray]]:
    """Fit a skew generalized secant hyperbolic distribution."""
    start = time.perf_counter()
    mean0 = float(np.mean(data))
    std0 = max(float(np.std(data, ddof=1)), 1e-3)
    sample_skew = float(stats.skew(data))
    skew_guess = math.exp(np.clip(sample_skew, -2.0, 2.0))
    skew_guess = min(max(skew_guess, 0.05), 20.0)
    theta0 = np.array([mean0, math.log(std0), 0.0, math.log(skew_guess)])

    def nll(theta: np.ndarray) -> float:
        mu, log_scale, t_param, skew_raw = theta
        scale = math.exp(log_scale)
        skew = math.exp(skew_raw)
        if not math.isfinite(scale) or scale <= 0.0 or skew <= 0.0:
            return np.inf
        pdf_vals = np.array(
            [sgsh_pdf(float((xi - mu) / scale), t_param, skew, SGSH_QUAD_LIMIT) for xi in data]
        ) / scale
        if np.any(~np.isfinite(pdf_vals)):
            return 1e20
        pdf_vals = np.maximum(pdf_vals, 1e-300)
        return -float(np.sum(np.log(pdf_vals)))

    bounds = [(None, None), (None, None), (-5.0, 5.0), (-6.0, 6.0)]
    init_list = [
        theta0,
        np.array([mean0, math.log(std0 * 0.75), -0.5, math.log(min(skew_guess * 0.7, 10.0))]),
        np.array([mean0, math.log(std0 * 1.25), 0.5, math.log(min(skew_guess * 1.3, 25.0))]),
        np.array([mean0, math.log(std0), -1.0, math.log(0.5)]),
        np.array([mean0, math.log(std0), 1.0, math.log(2.0)]),
    ]
    res = None
    last_message = "no optimization attempted"
    for guess in init_list:
        try:
            res = minimize(
                nll,
                guess,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 600},
            )
        except Exception as exc:
            last_message = str(exc)
            continue
        if res.success and math.isfinite(res.fun):
            break
        last_message = res.message if hasattr(res, "message") else "unknown failure"
    if res is None or not res.success:
        raise RuntimeError(f"SGSH fit failed: {last_message}")
    mu, log_scale, t_param, skew_raw = res.x
    scale = math.exp(log_scale)
    skew = math.exp(skew_raw)
    loglike = -res.fun

    def sgsh_cdf_val(x: float | np.ndarray) -> float | np.ndarray:
        def single_value(xi: float) -> float:
            if xi <= SGSH_CDF_LOWER:
                return 0.0
            val, _ = quad(
                lambda y: sgsh_pdf(float((y - mu) / scale), t_param, skew, SGSH_QUAD_LIMIT) / scale,
                SGSH_CDF_LOWER,
                xi,
                limit=SGSH_QUAD_LIMIT,
                epsabs=1e-6,
                epsrel=1e-6,
            )
            return max(0.0, min(1.0, val))

        arr = np.atleast_1d(x).astype(float)
        vals = np.array([single_value(xi) for xi in arr])
        if np.ndim(x) == 0:
            return float(vals[0])
        return vals

    ks_stat, ks_pvalue = stats.kstest(data, sgsh_cdf_val)
    fit_time = time.perf_counter() - start

    def pdf_func(grid: np.ndarray) -> np.ndarray:
        z = np.atleast_1d((grid - mu) / scale).astype(float)
        vals = np.array([sgsh_pdf(float(val), t_param, skew, SGSH_QUAD_LIMIT) for val in z]) / scale
        if np.ndim(grid) == 0:
            return np.array([vals[0]])
        return vals

    mean_d, std_d, skew_d, kurt_d = compute_density_moments(pdf_func, data)
    record = {
        "distribution": "sgsh",
        "scipy_name": None,
        "loglike": float(loglike),
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "params": (float(mu), float(scale), float(t_param), float(skew)),
        "n_params": 4,
        "fit_time": fit_time,
        "mean_density": mean_d,
        "std_density": std_d,
        "skew_density": skew_d,
        "kurt_density": kurt_d,
        "param_names": ["mu", "scale", "t", "skew"],
    }
    return record, pdf_func


CUSTOM_FITTERS["sgsh"] = fit_sgsh


def fit_nefghs(data: np.ndarray) -> tuple[dict[str, float | str], Callable[[np.ndarray], np.ndarray]]:
    """Fit an NEF-GHS distribution."""
    start = time.perf_counter()
    mean0 = float(np.mean(data))
    std0 = max(float(np.std(data, ddof=1)), 1e-3)
    theta0 = np.array([mean0, math.log(std0), math.log(1.0), 0.0])

    def nll(theta: np.ndarray) -> float:
        mu, log_scale, log_kappa, theta_raw = theta
        scale = math.exp(log_scale)
        kappa = math.exp(log_kappa) + 1e-6
        theta_param = 0.99 * math.tanh(theta_raw)
        if scale <= 0.0 or kappa <= 0.0 or abs(theta_param) >= 1.0:
            return np.inf
        pdf_vals = np.array(
            [nef_ghs_pdf(float(xi), kappa, theta_param, mu, scale) for xi in data]
        )
        if np.any(pdf_vals <= 0.0) or np.any(~np.isfinite(pdf_vals)):
            return np.inf
        return -float(np.sum(np.log(pdf_vals)))

    res = minimize(
        nll,
        theta0,
        method="L-BFGS-B",
        bounds=[(None, None), (None, None), (math.log(1e-6), None), (None, None)],
        options={"maxiter": 400},
    )
    if not res.success:
        raise RuntimeError(f"NEF-GHS fit failed: {res.message}")
    mu, log_scale, log_kappa, theta_raw = res.x
    scale = math.exp(log_scale)
    kappa = math.exp(log_kappa) + 1e-6
    theta_param = 0.99 * math.tanh(theta_raw)
    loglike = -res.fun

    def nefghs_cdf_val(x: float | np.ndarray) -> float | np.ndarray:
        def single_value(xi: float) -> float:
            tail = nef_ghs_tail_prob(xi, kappa, theta_param, mu, scale, NEF_GHS_LIMIT)
            return max(0.0, min(1.0, 1.0 - tail))

        arr = np.atleast_1d(x).astype(float)
        vals = np.array([single_value(xi) for xi in arr])
        if np.ndim(x) == 0:
            return float(vals[0])
        return vals

    ks_stat, ks_pvalue = stats.kstest(data, nefghs_cdf_val)
    fit_time = time.perf_counter() - start

    def pdf_func(grid: np.ndarray) -> np.ndarray:
        arr = np.atleast_1d(grid).astype(float)
        vals = np.array([nef_ghs_pdf(float(val), kappa, theta_param, mu, scale) for val in arr])
        if np.ndim(grid) == 0:
            return np.array([vals[0]])
        return vals

    mean_d, std_d, skew_d, kurt_d = compute_density_moments(pdf_func, data)
    record = {
        "distribution": "nefghs",
        "scipy_name": None,
        "loglike": float(loglike),
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "params": (float(mu), float(scale), float(kappa), float(theta_param)),
        "n_params": 4,
        "fit_time": fit_time,
        "mean_density": mean_d,
        "std_density": std_d,
        "skew_density": skew_d,
        "kurt_density": kurt_d,
        "param_names": ["mu", "scale", "kappa", "theta"],
    }
    return record, pdf_func


CUSTOM_FITTERS["nefghs"] = fit_nefghs


def fit_vg(data: np.ndarray) -> tuple[dict[str, float | str], Callable[[np.ndarray], np.ndarray]]:
    """Fit a variance gamma distribution."""
    start = time.perf_counter()
    mean0 = float(np.mean(data))
    std0 = max(float(np.std(data, ddof=1)), 1e-3)
    theta0 = np.array([mean0, 0.0, math.log(std0 * 0.5), math.log(1.0), math.log(1.0)])

    def compute_norm(mu, theta_param, sigma, shape, scale):
        g_mean = shape * scale
        g_var = shape * (scale * scale)
        approx_var = sigma * sigma * g_mean + theta_param * theta_param * g_var
        span = max(10.0, 8.0 * math.sqrt(max(approx_var, 1e-6)))
        grid = np.linspace(mu - span, mu + span, 2000)
        norm_vals = np.array([vg_pdf(float(xi), mu, theta_param, sigma, shape, scale) for xi in grid])
        norm = np.trapezoid(norm_vals, grid)
        return norm

    def nll(theta: np.ndarray) -> float:
        mu, theta_param, log_sigma, log_shape, log_scale = theta
        sigma = math.exp(log_sigma)
        shape = math.exp(log_shape) + 1e-6
        scale = math.exp(log_scale) + 1e-6
        if sigma <= 0.0:
            return np.inf
        norm = compute_norm(mu, theta_param, sigma, shape, scale)
        if norm <= 0.0 or not math.isfinite(norm):
            return np.inf
        raw_vals = np.array([vg_pdf(float(xi), mu, theta_param, sigma, shape, scale) for xi in data])
        pdf_vals = raw_vals / norm
        if np.any(pdf_vals <= 0.0) or np.any(~np.isfinite(pdf_vals)):
            return np.inf
        return -float(np.sum(np.log(pdf_vals)))

    bounds = [
        (None, None),
        (None, None),
        (math.log(1e-3), math.log(50.0)),
        (math.log(1e-3), math.log(50.0)),
        (math.log(1e-3), math.log(50.0)),
    ]
    res = minimize(
        nll,
        theta0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 600},
    )
    if not res.success:
        raise RuntimeError(f"VG fit failed: {res.message}")
    mu, theta_param, log_sigma, log_shape, log_scale = res.x
    sigma = math.exp(log_sigma)
    shape = math.exp(log_shape) + 1e-6
    scale = math.exp(log_scale) + 1e-6
    loglike = -res.fun
    norm_final = compute_norm(mu, theta_param, sigma, shape, scale)
    if norm_final <= 0.0 or not math.isfinite(norm_final):
        raise RuntimeError("VG normalization failed")

    def vg_cdf_val(x: float | np.ndarray) -> float | np.ndarray:
        def single_value(xi: float) -> float:
            if xi <= VG_CDF_LOWER:
                return 0.0
            val, _ = quad(
                lambda y: vg_pdf(float(y), mu, theta_param, sigma, shape, scale) / norm_final,
                VG_CDF_LOWER,
                xi,
                limit=VG_QUAD_LIMIT,
                epsabs=1e-6,
                epsrel=1e-6,
            )
            return max(0.0, min(1.0, val))

        arr = np.atleast_1d(x).astype(float)
        vals = np.array([single_value(xi) for xi in arr])
        if np.ndim(x) == 0:
            return float(vals[0])
        return vals

    ks_stat, ks_pvalue = stats.kstest(data, vg_cdf_val)
    fit_time = time.perf_counter() - start

    def pdf_func(grid: np.ndarray) -> np.ndarray:
        arr = np.atleast_1d(grid).astype(float)
        vals = np.array([vg_pdf(float(val), mu, theta_param, sigma, shape, scale) for val in arr]) / norm_final
        if np.ndim(grid) == 0:
            return np.array([vals[0]])
        return vals

    mean_d, std_d, skew_d, kurt_d = compute_density_moments(pdf_func, data)
    record = {
        "distribution": "vg",
        "scipy_name": None,
        "loglike": float(loglike),
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "params": (float(mu), float(theta_param), float(sigma), float(shape), float(scale)),
        "n_params": 5,
        "fit_time": fit_time,
        "mean_density": mean_d,
        "std_density": std_d,
        "skew_density": skew_d,
        "kurt_density": kurt_d,
        "param_names": ["mu", "theta", "sigma", "shape", "scale"],
    }
    return record, pdf_func


CUSTOM_FITTERS["vg"] = fit_vg


def fit_skew_t(data: np.ndarray) -> tuple[dict[str, float | str], Callable[[np.ndarray], np.ndarray]]:
    """Fit the Jones-Faddy skew t distribution via constrained optimization."""
    start = time.perf_counter()
    mean0 = float(np.mean(data))
    std0 = max(float(np.std(data, ddof=1)), 1e-3)
    theta0 = np.array([math.log(2.0), math.log(2.0), math.log(std0), mean0])

    def nll(theta: np.ndarray) -> float:
        log_a, log_b, log_scale, loc = theta
        a = math.exp(log_a) + 1e-6
        b = math.exp(log_b) + 1e-6
        scale = math.exp(log_scale)
        if scale <= 0.0:
            return np.inf
        pdf_vals = stats.jf_skew_t.pdf(data, a, b, loc=loc, scale=scale)
        if np.any(pdf_vals <= 0.0) or np.any(~np.isfinite(pdf_vals)):
            return np.inf
        return -float(np.sum(np.log(pdf_vals)))

    res = minimize(
        nll,
        theta0,
        method="L-BFGS-B",
        bounds=[(math.log(1e-6), None), (math.log(1e-6), None), (math.log(1e-6), None), (None, None)],
        options={"maxiter": 400},
    )
    if not res.success:
        raise RuntimeError(f"skew_t fit failed: {res.message}")
    log_a, log_b, log_scale, loc = res.x
    a = math.exp(log_a) + 1e-6
    b = math.exp(log_b) + 1e-6
    scale = math.exp(log_scale)
    loglike = -res.fun

    def skew_t_cdf_val(x: float | np.ndarray) -> float | np.ndarray:
        arr = np.atleast_1d(x).astype(float)
        vals = stats.jf_skew_t.cdf(arr, a, b, loc=loc, scale=scale)
        if np.ndim(x) == 0:
            return float(vals[0])
        return vals

    ks_stat, ks_pvalue = stats.kstest(data, skew_t_cdf_val)
    fit_time = time.perf_counter() - start

    def pdf_func(grid: np.ndarray) -> np.ndarray:
        arr = np.atleast_1d(grid).astype(float)
        vals = stats.jf_skew_t.pdf(arr, a, b, loc=loc, scale=scale)
        if np.ndim(grid) == 0:
            return np.array([vals[0]])
        return vals

    mean_d, std_d, skew_d, kurt_d = compute_density_moments(pdf_func, data)
    record = {
        "distribution": "skew_t",
        "scipy_name": None,
        "loglike": float(loglike),
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "params": (a, b, loc, scale),
        "n_params": 4,
        "fit_time": fit_time,
        "mean_density": mean_d,
        "std_density": std_d,
        "skew_density": skew_d,
        "kurt_density": kurt_d,
        "param_names": ["a", "b", "loc", "scale"],
    }
    return record, pdf_func


CUSTOM_FITTERS["skew_t"] = fit_skew_t


def plot_densities(
    data: np.ndarray,
    components: list[tuple[float, float, float]],
    pdf_entries: list[tuple[str, Callable[[np.ndarray], np.ndarray]]],
    outfile: Optional[str],
) -> None:
    """Plot the empirical and fitted densities."""
    if not pdf_entries:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping density plot")
        return
    x_min = float(np.min(data)) - 1.0
    x_max = float(np.max(data)) + 1.0
    grid = np.linspace(x_min, x_max, 1000)
    true_pdf = np.zeros_like(grid)
    for weight, mean, std in components:
        true_pdf += weight * stats.norm.pdf(grid, loc=mean, scale=std)
    true_pdf /= sum(w for w, _, _ in components)
    plt.figure(figsize=(9, 5))
    plt.plot(grid, true_pdf, label="true mixture", linewidth=2.0, color="black")
    for label, pdf_func in pdf_entries:
        try:
            pdf_vals = pdf_func(grid)
            plt.plot(grid, pdf_vals, label=label)
        except Exception:
            continue
    plt.title("Fitted densities vs true mixture")
    plt.xlabel("x")
    plt.ylabel("pdf")
    plt.legend()
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile)
        plt.close()
    else:
        plt.show()

