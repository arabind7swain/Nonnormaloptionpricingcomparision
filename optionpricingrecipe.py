import math
import cmath
from typing import List, Literal, Mapping, Optional, Tuple

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import beta as beta_fn, betainc
from scipy.stats import norm, tukeylambda

from distributions import (
    CGMY_U_MAX,
    NEF_GHS_MGF_LIMIT,
    champernowne_params,
    champernowne_pdf,
    champernowne_tail,
    champernowne_tilted_tail,
    ged_params,
    ged_tail_prob,
    ged_tilted_tail,
    gh_params,
    gh_pdf,
    gh_base_stats,
    gh_tail_prob,
    gh_tilted_tail,
    hyperbolic_params,
    hyperbolic_pdf,
    hyperbolic_tail_prob,
    hyperbolic_tilted_tail,
    cgmy_params,
    cgmy_characteristic,
    cgmy_mgf,
    cgmy_pdf,
    johnson_su_params,
    johnson_su_pdf,
    johnson_su_tail_prob,
    johnson_su_tilted_tail,
    hypsecant_params,
    hypsecant_pdf_val,
    laplace_params,
    laplace_tail_prob,
    laplace_tilted_tail,
    logistic_params,
    logistic_pdf_val,
    logistic_tail_prob,
    gsh_pdf,
    sgsh_pdf,
    gsh_params,
    sgsh_params,
    sgsh_base_moments,
    sgsh_base_high_moments,
    nig_delta,
    nig_pdf,
    nig_tail_prob,
    nig_tilted_tail,
    normal_params,
    nef_ghs_params,
    nef_ghs_base_moments,
    nef_ghs_pdf,
    nef_ghs_mgf,
    tukey_lambda_params,
    tukey_lambda_moment_values,
    tukey_lambda_pdf,
    tukey_lambda_tail_prob,
    tukey_lambda_tilted_tail,
    vg_call_expectation,
    vg_pdf,
)

LOG_XMAX = 40.0


def _integration_edges(vol_sqrt_t: float) -> List[float]:
    """Return sorted integration edges concentrating near the origin."""
    coarse_breaks = [-20.0, -10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0, 20.0]
    fine_span = max(0.25, 8.0 * max(vol_sqrt_t, 0.0))
    fine_breaks = [i * fine_span for i in range(-10, 11)]
    edges = [-LOG_XMAX]
    edges.extend(val for val in coarse_breaks if -LOG_XMAX < val < LOG_XMAX)
    edges.extend(val for val in fine_breaks if -LOG_XMAX < val < LOG_XMAX)
    edges.append(LOG_XMAX)
    edges = sorted(set(edges))
    return edges


def bs_call_price(s0: float, k: float, t: float, r: float, q: float, vol: float) -> float:
    """Black-Scholes call price for given volatility."""
    if vol <= 0.0 or t <= 0.0:
        return max(s0 * math.exp(-q * t) - k * math.exp(-r * t), 0.0)
    vsqrt = vol * math.sqrt(t)
    d1 = (math.log(s0 / k) + (r - q + 0.5 * vol * vol) * t) / vsqrt
    d2 = d1 - vsqrt
    return s0 * math.exp(-q * t) * norm.cdf(d1) - k * math.exp(-r * t) * norm.cdf(d2)


def implied_vol_from_call(price: float, s0: float, k: float, r: float, q: float, t: float) -> float:
    """Invert the BS formula to recover implied volatility."""
    intrinsic = max(s0 * math.exp(-q * t) - k * math.exp(-r * t), 0.0)
    upper = s0 * math.exp(-q * t)
    target = min(max(price, intrinsic + 1e-12), upper - 1e-12)

    def f(vol: float) -> float:
        return bs_call_price(s0, k, t, r, q, vol) - target

    lo = 1e-8
    hi = 5.0
    flo = f(lo)
    fhi = f(hi)
    if flo > 0.0:
        return lo
    if fhi < 0.0:
        raise RuntimeError("could not bracket implied volatility")
    return brentq(f, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=200)


def plot_terminal_distributions(
    s0: float,
    sigma: float,
    r: float,
    q: float,
    t: float,
    dist_columns: List[Tuple[str, str, Optional[dict[str, float]]]],
    quad_limit: int,
    outfile: Optional[str] = None,
) -> None:
    """Plot all terminal price distributions on a single graph."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping distribution plot")
        return

    sqrt_t = math.sqrt(t)
    x_grid = np.linspace(-1.0, 1.0, 800)
    s_grid = s0 * np.exp(x_grid)

    plt.figure()

    for label, dist_name, dist_params in dist_columns:
        try:
            if dist_name == "normal":
                mu, vol = normal_params(sigma, t, r, q)
                pdf_x = norm.pdf((x_grid - mu) / vol) / vol
            elif dist_name == "hypsecant":
                mu, b = hypsecant_params(sigma, sqrt_t, r, q, t)
                z = (x_grid - mu) / b
                pdf_x = (0.5 / b) / np.cosh(0.5 * math.pi * z)
            elif dist_name == "logistic":
                mu, scale = logistic_params(sigma, sqrt_t, r, q, t)
                z = (x_grid - mu) / scale
                ez = np.exp(-z)
                pdf_x = ez / (scale * (1.0 + ez) ** 2)
            elif dist_name == "laplace":
                mu, scale = laplace_params(sigma, sqrt_t, r, q, t)
                pdf_x = (0.5 / scale) * np.exp(-np.abs(x_grid - mu) / scale)
            elif dist_name == "johnsonsu":
                if not dist_params:
                    raise ValueError("johnsonsu requires dist_params")
                a = float(dist_params["a"])
                b = float(dist_params["b"])
                loc, scale = johnson_su_params(a, b, sigma, sqrt_t, r, q, t, quad_limit)
                pdf_vals = [
                    johnson_su_pdf(float(val), a, b, loc, scale) for val in x_grid
                ]
                pdf_x = np.array(pdf_vals)
            elif dist_name == "tukeylambda":
                if not dist_params:
                    raise ValueError("tukeylambda requires dist_params")
                lam = float(dist_params["lam"])
                loc, scale = tukey_lambda_params(lam, sigma, sqrt_t, r, q, t, quad_limit)
                pdf_vals = [
                    tukey_lambda_pdf(float(val), lam, loc, scale) for val in x_grid
                ]
                pdf_x = np.array(pdf_vals)
            elif dist_name == "hyperbolic":
                if not dist_params:
                    raise ValueError("hyperbolic requires dist_params")
                alpha = float(dist_params["alpha"])
                beta = float(dist_params["beta"])
                mu, delta, alpha_eff, beta_eff = hyperbolic_params(alpha, beta, sigma, r, q, t)
                pdf_vals = [
                    hyperbolic_pdf(float(val), alpha_eff, beta_eff, delta, mu) for val in x_grid
                ]
                pdf_x = np.array(pdf_vals)
            elif dist_name == "gsh":
                if not dist_params:
                    raise ValueError("gsh requires dist_params")
                param_t = float(dist_params["t"])
                mu_g, scale_g = gsh_params(param_t, sigma, sqrt_t, r, q, t, quad_limit)
                pdf_vals = [
                    gsh_pdf((float(val) - mu_g) / scale_g, param_t, quad_limit) / scale_g
                    for val in x_grid
                ]
                pdf_x = np.array(pdf_vals)
            elif dist_name == "sgsh":
                if not dist_params:
                    raise ValueError("sgsh requires dist_params")
                param_t = float(dist_params["t"])
                skew_val = float(dist_params["skew"])
                mu_s, scale_s = sgsh_params(param_t, skew_val, sigma, sqrt_t, r, q, t, quad_limit)
                pdf_vals = [
                    sgsh_pdf(
                        (float(val) - mu_s) / scale_s,
                        param_t,
                        skew_val,
                        quad_limit,
                    )
                    / scale_s
                    for val in x_grid
                ]
                pdf_x = np.array(pdf_vals)
            elif dist_name == "nefghs":
                if not dist_params:
                    raise ValueError("nefghs requires dist_params")
                if "kappa" not in dist_params:
                    raise ValueError("nefghs requires dist_params with key kappa")
                kappa = float(dist_params["kappa"])
                theta = _nef_theta(dist_params)
                mu, scale, theta = nef_ghs_params(kappa, theta, sigma, sqrt_t, r, q, t)
                pdf_vals = [
                    nef_ghs_pdf(float(val), kappa, theta, mu, scale) for val in x_grid
                ]
                pdf_x = np.array(pdf_vals)
            elif dist_name == "gh":
                if not dist_params:
                    raise ValueError("gh requires dist_params")
                lam = float(dist_params["lam"])
                alpha = float(dist_params["alpha"])
                beta = float(dist_params["beta"])
                loc, scale = gh_params(lam, alpha, beta, sigma, sqrt_t, r, q, t, quad_limit)
                pdf_vals = [
                    gh_pdf(float(val), lam, alpha, beta, loc, scale) for val in x_grid
                ]
                pdf_x = np.array(pdf_vals)
            elif dist_name == "cgmy":
                if not dist_params:
                    raise ValueError("cgmy requires dist_params")
                C_val = float(dist_params["C"])
                G_val = float(dist_params["G"])
                M_val = float(dist_params["M"])
                Y_val = float(dist_params["Y"])
                params_cgmy = cgmy_params(C_val, G_val, M_val, Y_val, sigma, sqrt_t, r, q, t)
                pdf_vals = [
                    cgmy_pdf(float(val), params_cgmy, quad_limit) for val in x_grid
                ]
                pdf_x = np.array(pdf_vals)
            elif dist_name == "champernowne":
                d = float(dist_params["d"]) if dist_params else 0.0
                mu, scale, norm_const = champernowne_params(d, sigma, sqrt_t, r, q, t, quad_limit)
                z = (x_grid - mu) / scale
                pdf_vals = [champernowne_pdf(float(val), d, norm_const) for val in z]
                pdf_x = np.array(pdf_vals) / scale
            elif dist_name == "ged":
                power = float(dist_params["p"]) if dist_params else 2.0
                mu, scale, norm_const = ged_params(power, sigma, sqrt_t, r, q, t, quad_limit)
                pdf_x = norm_const * np.exp(-np.power(np.abs(x_grid - mu) / scale, power))
            elif dist_name == "nig":
                alpha = float(dist_params["alpha"])
                beta = float(dist_params["beta"])
                variance = sigma * sigma * t
                delta = nig_delta(variance, alpha, beta)
                log_mgf = delta * (
                    math.sqrt(alpha * alpha - beta * beta)
                    - math.sqrt(alpha * alpha - (beta + 1.0) ** 2)
                )
                mu = (r - q) * t - log_mgf
                pdf_vals = [nig_pdf(float(val), alpha, beta, delta, mu) for val in x_grid]
                pdf_x = np.array(pdf_vals)
            elif dist_name == "vg":
                theta = float(dist_params["theta"])
                nu = float(dist_params["nu"])
                shape = t / nu
                scale = nu
                sigma_b_sq = sigma * sigma - nu * theta * theta
                if sigma_b_sq <= 0.0:
                    raise ValueError("vg requires sigma^2 > nu * theta^2 to match variance")
                sigma_b = math.sqrt(sigma_b_sq)
                arg = 1.0 - nu * (theta + 0.5 * sigma_b_sq)
                if arg <= 0.0:
                    raise ValueError("vg requires 1 - nu*(theta + 0.5*sigma_b^2) > 0")
                mu = (r - q) * t + (t / nu) * math.log(arg)
                pdf_vals = [vg_pdf(float(val), mu, theta, sigma_b, shape, scale) for val in x_grid]
                pdf_x = np.array(pdf_vals)
            else:
                continue
        except Exception as exc:  # pragma: no cover
            print(f"could not plot {label}: {exc}")
            continue

        mass_x = np.trapz(pdf_x, x_grid)
        if mass_x > 0.0:
            pdf_x = pdf_x / mass_x

        pdf_s = pdf_x / s_grid
        plt.plot(s_grid, pdf_s, label=label)

    plt.title("Terminal price distributions")
    plt.xlabel("S_T")
    plt.ylabel("pdf")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile)
        plt.close()
    else:
        plt.show()


def plot_implied_vols(
    strikes: List[float],
    vol_values: List[List[float]],
    dist_labels: List[str],
    outfile: Optional[str] = None,
) -> None:
    """Plot implied vol curves for each distribution."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping implied vol plot")
        return

    x = np.array(strikes, dtype=float)
    plt.figure()

    for col_idx, label in enumerate(dist_labels):
        vols = [row[col_idx] for row in vol_values]
        plt.plot(x, vols, label=label)
    plt.title("Implied volatility curves")
    plt.xlabel("strike")
    plt.ylabel("implied vol")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile)
        plt.close()
    else:
        plt.show()


def _nef_theta(dist_params: Mapping[str, float]) -> float:
    """Return canonical theta for NEF-GHS parameters."""
    if "theta" in dist_params:
        theta = float(dist_params["theta"])
    elif "lam" in dist_params:
        theta = math.atan(float(dist_params["lam"]))
    else:
        raise ValueError('nefghs requires dist_params with key "theta" or "lam"')
    if abs(theta) >= 1.0:
        raise ValueError("nefghs canonical parameter theta must satisfy |theta| < 1")
    return theta


def _log_pdf_factory(
    dist_name: str,
    dist_params: Optional[Mapping[str, float]],
    sigma: float,
    sqrt_t: float,
    r: float,
    q: float,
    t: float,
    quad_limit: int,
):
    """Return a callable pdf(x) for log returns."""
    if dist_name == "normal":
        mu, vol = normal_params(sigma, t, r, q)
        return lambda x: norm.pdf((x - mu) / vol) / vol
    if dist_name == "hypsecant":
        mu, b = hypsecant_params(sigma, sqrt_t, r, q, t)
        return lambda x: hypsecant_pdf_val(x, mu, b)
    if dist_name == "logistic":
        mu, scale = logistic_params(sigma, sqrt_t, r, q, t)
        return lambda x: logistic_pdf_val(x, mu, scale)
    if dist_name == "laplace":
        mu, scale = laplace_params(sigma, sqrt_t, r, q, t)
        return lambda x: 0.5 / scale * math.exp(-abs(x - mu) / scale)
    if dist_name == "johnsonsu":
        if not dist_params or "a" not in dist_params or "b" not in dist_params:
            raise ValueError("johnsonsu requires dist_params with keys a and b")
        a = float(dist_params["a"])
        b = float(dist_params["b"])
        loc, scale = johnson_su_params(a, b, sigma, sqrt_t, r, q, t, quad_limit)
        return lambda x: johnson_su_pdf(x, a, b, loc, scale)
    if dist_name == "tukeylambda":
        if not dist_params or "lam" not in dist_params:
            raise ValueError("tukeylambda requires dist_params with key lam")
        lam = float(dist_params["lam"])
        loc, scale = tukey_lambda_params(lam, sigma, sqrt_t, r, q, t, quad_limit)
        return lambda x: tukey_lambda_pdf(x, lam, loc, scale)
    if dist_name == "hyperbolic":
        if not dist_params or "alpha" not in dist_params or "beta" not in dist_params:
            raise ValueError("hyperbolic requires dist_params with keys alpha and beta")
        alpha = float(dist_params["alpha"])
        beta = float(dist_params["beta"])
        mu, delta, alpha_eff, beta_eff = hyperbolic_params(alpha, beta, sigma, r, q, t)
        return lambda x: hyperbolic_pdf(x, alpha_eff, beta_eff, delta, mu)
    if dist_name == "gsh":
        if not dist_params or "t" not in dist_params:
            raise ValueError("gsh requires dist_params with key t")
        param_t = float(dist_params["t"])
        mu, scale = gsh_params(param_t, sigma, sqrt_t, r, q, t, quad_limit)
        return lambda x: gsh_pdf((x - mu) / scale, param_t, quad_limit) / scale
    if dist_name == "sgsh":
        if not dist_params or "t" not in dist_params or "skew" not in dist_params:
            raise ValueError("sgsh requires dist_params with keys t and skew")
        param_t = float(dist_params["t"])
        skew = float(dist_params["skew"])
        mu, scale = sgsh_params(param_t, skew, sigma, sqrt_t, r, q, t, quad_limit)
        return lambda x: sgsh_pdf((x - mu) / scale, param_t, skew, quad_limit) / scale
    if dist_name == "nefghs":
        if not dist_params or "kappa" not in dist_params:
            raise ValueError("nefghs requires dist_params with key kappa")
        kappa = float(dist_params["kappa"])
        theta = _nef_theta(dist_params)
        mu, scale, theta = nef_ghs_params(kappa, theta, sigma, sqrt_t, r, q, t)
        return lambda x: nef_ghs_pdf(x, kappa, theta, mu, scale)
    if dist_name == "gh":
        if not dist_params:
            raise ValueError("generalized hyperbolic requires dist_params")
        lam = float(dist_params["lam"])
        alpha = float(dist_params["alpha"])
        beta = float(dist_params["beta"])
        loc, scale = gh_params(lam, alpha, beta, sigma, sqrt_t, r, q, t, quad_limit)
        return lambda x: gh_pdf(x, lam, alpha, beta, loc, scale)
    if dist_name == "champernowne":
        if not dist_params:
            raise ValueError("champernowne requires dist_params")
        d = float(dist_params["d"])
        mu, scale, norm_const = champernowne_params(d, sigma, sqrt_t, r, q, t, quad_limit)
        return lambda x: champernowne_pdf((x - mu) / scale, d, norm_const) / scale
    if dist_name == "cgmy":
        if not dist_params:
            raise ValueError("cgmy requires dist_params")
        C_val = float(dist_params["C"])
        G_val = float(dist_params["G"])
        M_val = float(dist_params["M"])
        Y_val = float(dist_params["Y"])
        params_cgmy = cgmy_params(C_val, G_val, M_val, Y_val, sigma, sqrt_t, r, q, t)
        return lambda x: cgmy_pdf(x, params_cgmy, quad_limit)
    if dist_name == "ged":
        if not dist_params:
            raise ValueError("ged requires dist_params")
        power = float(dist_params["p"])
        mu, scale, norm_const = ged_params(power, sigma, sqrt_t, r, q, t, quad_limit)
        return lambda x: norm_const * math.exp(-abs((x - mu) / scale) ** power)
    if dist_name == "nig":
        if not dist_params:
            raise ValueError("nig requires dist_params")
        alpha = float(dist_params["alpha"])
        beta = float(dist_params["beta"])
        variance = sigma * sigma * t
        delta = nig_delta(variance, alpha, beta)
        log_mgf = delta * (
            math.sqrt(alpha * alpha - beta * beta)
            - math.sqrt(alpha * alpha - (beta + 1.0) ** 2)
        )
        mu = (r - q) * t - log_mgf
        return lambda x: nig_pdf(x, alpha, beta, delta, mu)
    if dist_name == "vg":
        if not dist_params:
            raise ValueError("vg requires dist_params")
        theta = float(dist_params["theta"])
        nu = float(dist_params["nu"])
        shape = t / nu
        scale = nu
        sigma_sq_target = sigma * sigma
        sigma_b_sq = sigma_sq_target - nu * theta * theta
        if sigma_b_sq <= 0.0:
            raise ValueError("vg requires sigma^2 > nu * theta^2 to match variance")
        sigma_b = math.sqrt(sigma_b_sq)
        arg = 1.0 - nu * (theta + 0.5 * sigma_b_sq)
        if arg <= 0.0:
            raise ValueError("vg requires 1 - nu*(theta + 0.5*sigma_b^2) > 0")
        mu = (r - q) * t + (t / nu) * math.log(arg)
        return lambda x: vg_pdf(x, mu, theta, sigma_b, shape, scale)
    raise ValueError(f"unsupported dist {dist_name}")


def log_return_stats(
    dist_name: str,
    dist_params: Optional[Mapping[str, float]],
    sigma: float,
    r: float,
    q: float,
    t: float,
    quad_limit: int,
) -> Tuple[float, float, float, float]:
    """Compute mean/std/skew/kurtosis of the log-price distribution."""
    # Closed form cumulants for VG keep the existing fast path.
    if dist_name == "vg":
        if not dist_params:
            raise ValueError("vg requires dist_params")
        theta = float(dist_params["theta"])
        nu = float(dist_params["nu"])
        sigma_sq_target = sigma * sigma
        sigma_b_sq = sigma_sq_target - nu * theta * theta
        if sigma_b_sq <= 0.0:
            raise ValueError("vg requires sigma^2 > nu * theta^2 to match variance")
        arg = 1.0 - nu * (theta + 0.5 * sigma_b_sq)
        if arg <= 0.0:
            raise ValueError("vg requires 1 - nu*(theta + 0.5*sigma_b^2) > 0")
        mu_adj = (r - q) * t + (t / nu) * math.log(arg)
        mean = mu_adj + t * theta
        variance = t * (sigma_b_sq + nu * theta * theta)
        std = math.sqrt(variance)
        kappa3 = t * theta * nu * (3.0 * sigma_b_sq + 2.0 * theta * theta * nu)
        kappa4 = t * (
            3.0 * sigma_b_sq * sigma_b_sq * nu
            + 12.0 * sigma_b_sq * theta * theta * nu * nu
            + 6.0 * theta**4 * nu**3
        )
        skew = kappa3 / (std ** 3) if std > 0.0 else 0.0
        kurt = 3.0 + (kappa4 / (variance * variance)) if variance > 0.0 else 3.0
        return mean, std, skew, kurt
    if dist_name == "cgmy":
        if not dist_params:
            raise ValueError("cgmy requires dist_params")
        C_val = float(dist_params["C"])
        G_val = float(dist_params["G"])
        M_val = float(dist_params["M"])
        Y_val = float(dist_params["Y"])
        sqrt_t = math.sqrt(t)
        params_cgmy = cgmy_params(C_val, G_val, M_val, Y_val, sigma, sqrt_t, r, q, t)
        c_val, g_val, m_val, y_val, scale, mu_shift, gamma_neg, t_span = params_cgmy
        coeff = c_val * gamma_neg * t_span
        kappa1 = coeff * scale * y_val * (g_val ** (y_val - 1.0) - m_val ** (y_val - 1.0))
        kappa2 = coeff * (scale**2) * y_val * (y_val - 1.0) * (
            m_val ** (y_val - 2.0) + g_val ** (y_val - 2.0)
        )
        kappa3 = coeff * (scale**3) * y_val * (y_val - 1.0) * (y_val - 2.0) * (
            g_val ** (y_val - 3.0) - m_val ** (y_val - 3.0)
        )
        kappa4 = coeff * (scale**4) * y_val * (y_val - 1.0) * (y_val - 2.0) * (y_val - 3.0) * (
            m_val ** (y_val - 4.0) + g_val ** (y_val - 4.0)
        )
        mean = mu_shift + kappa1
        variance = kappa2
        std = math.sqrt(max(variance, 0.0))
        if std == 0.0:
            return mean, 0.0, 0.0, 3.0
        skew = kappa3 / (std**3)
        kurt = 3.0 + (kappa4 / (variance * variance))
        return mean, std, skew, kurt
    if dist_name == "tukeylambda":
        if not dist_params or "lam" not in dist_params:
            raise ValueError("tukeylambda requires dist_params")
        lam = float(dist_params["lam"])
        sqrt_t = math.sqrt(t)
        loc, scale = tukey_lambda_params(lam, sigma, sqrt_t, r, q, t, quad_limit)
        mean_b, var_b, skew_b, kurt_b = tukey_lambda_moment_values(lam)
        if var_b <= 0.0:
            raise ValueError("tukeylambda variance is not finite; adjust lambda")
        mean = loc + scale * mean_b
        variance = (scale * scale) * var_b
        std = math.sqrt(max(variance, 0.0))
        if std == 0.0:
            return mean, 0.0, 0.0, 3.0
        return mean, std, skew_b, kurt_b
    if dist_name == "gh":
        if not dist_params:
            raise ValueError("generalized hyperbolic requires dist_params")
        lam = float(dist_params["lam"])
        alpha = float(dist_params["alpha"])
        beta = float(dist_params["beta"])
        sqrt_t = math.sqrt(t)
        loc, scale = gh_params(lam, alpha, beta, sigma, sqrt_t, r, q, t, quad_limit)
        mean_b, var_b, skew_b, kurt_b = gh_base_stats(lam, alpha, beta)
        mean = loc + scale * mean_b
        variance = (scale * scale) * var_b
        std = math.sqrt(max(variance, 0.0))
        if std == 0.0:
            return mean, 0.0, 0.0, 3.0
        return mean, std, skew_b, kurt_b
    if dist_name == "nefghs":
        if not dist_params or "kappa" not in dist_params:
            raise ValueError("nefghs requires dist_params with key kappa")
        kappa = float(dist_params["kappa"])
        theta = _nef_theta(dist_params)
        sqrt_t = math.sqrt(t)
        mu, scale, theta = nef_ghs_params(kappa, theta, sigma, sqrt_t, r, q, t)
        m1, m2, m3, m4 = nef_ghs_base_moments(theta, kappa, NEF_GHS_MGF_LIMIT)
        raw_var = m2 - m1 * m1
        variance = (scale * scale) * raw_var
        variance = max(variance, 0.0)
        mean = mu + scale * m1
        std = math.sqrt(variance)
        if std == 0.0:
            return mean, 0.0, 0.0, 3.0
        third_central = m3 - 3.0 * m2 * m1 + 2.0 * (m1**3)
        fourth_central = m4 - 4.0 * m3 * m1 + 6.0 * m2 * (m1**2) - 3.0 * (m1**4)
        skew = (scale**3 * third_central) / (std**3)
        kurt = (scale**4 * fourth_central) / (variance * variance)
        return mean, std, skew, kurt
    if dist_name == "sgsh":
        if not dist_params or "t" not in dist_params or "skew" not in dist_params:
            raise ValueError("sgsh requires dist_params with keys t and skew")
        param_t = float(dist_params["t"])
        skew_param = float(dist_params["skew"])
        sqrt_t = math.sqrt(t)
        mu, scale = sgsh_params(param_t, skew_param, sigma, sqrt_t, r, q, t, quad_limit)
        mean_base, var_base = sgsh_base_moments(param_t, skew_param, quad_limit)
        third_base, fourth_base = sgsh_base_high_moments(param_t, skew_param, quad_limit)
        mean = mu + scale * mean_base
        variance = (scale * scale) * var_base
        std = math.sqrt(max(variance, 0.0))
        if std == 0.0:
            return mean, 0.0, 0.0, 3.0
        third = (scale ** 3) * third_base
        fourth = (scale ** 4) * fourth_base
        skew = third / (std ** 3)
        kurt = fourth / (variance * variance)
        return mean, std, skew, kurt

    sqrt_t = math.sqrt(t)
    pdf = _log_pdf_factory(dist_name, dist_params, sigma, sqrt_t, r, q, t, quad_limit)

    def moment(power: int) -> float:
        """Return E[X**power] for the log return X."""
        integrand = lambda x: (x**power) * pdf(x)
        edges = _integration_edges(sigma * sqrt_t)
        per_segment_limit = max(50, quad_limit // max(len(edges) - 1, 1))
        total = 0.0
        for start, end in zip(edges, edges[1:]):
            if end <= start:
                continue
            seg_val, _ = quad(
                integrand,
                start,
                end,
                limit=per_segment_limit,
                epsabs=1e-7,
                epsrel=1e-7,
            )
            total += seg_val
        return total

    m1 = moment(1)
    m2 = moment(2)
    var = m2 - m1 * m1
    std = math.sqrt(max(var, 0.0))
    if std == 0.0:
        return m1, 0.0, 0.0, 3.0
    m3 = moment(3)
    m4 = moment(4)
    skew = (m3 - 3.0 * m2 * m1 + 2.0 * m1**3) / (std**3)
    kurt = (m4 - 4.0 * m3 * m1 + 6.0 * m2 * m1 * m1 - 3.0 * m1**4) / (var * var)
    return m1, std, skew, kurt


def terminal_price_stats(
    dist_name: str,
    dist_params: Optional[Mapping[str, float]],
    s0: float,
    sigma: float,
    r: float,
    q: float,
    t: float,
    quad_limit: int,
) -> Tuple[float, float, float, float]:
    """Compute mean/std/skew/kurtosis of the terminal price S_T."""
    def moment_expect(power: float) -> float:
        """Return E[exp(power * X)] with X = log(S_T/S_0)."""
        if dist_name == "vg":
            if not dist_params:
                raise ValueError("vg requires dist_params")
            theta = float(dist_params["theta"])
            nu = float(dist_params["nu"])
            sigma_sq_target = sigma * sigma
            sigma_b_sq = sigma_sq_target - nu * theta * theta
            if sigma_b_sq <= 0.0:
                raise ValueError("vg requires sigma^2 > nu * theta^2 to match variance")
            denom = 1.0 - theta * nu * power - 0.5 * sigma_b_sq * nu * power * power
            if denom <= 0.0:
                raise ValueError("vg requires 1 - nu*(theta*u + 0.5*sigma_b^2*u^2) > 0")
            arg = 1.0 - nu * (theta + 0.5 * sigma_b_sq)
            if arg <= 0.0:
                raise ValueError("vg requires 1 - nu*(theta + 0.5*sigma_b^2) > 0")
            mu_adj = (r - q) * t + (t / nu) * math.log(arg)
            return math.exp(mu_adj * power) * (denom ** (-t / nu))
        if dist_name == "nefghs":
            if not dist_params or "kappa" not in dist_params:
                raise ValueError("nefghs requires dist_params with key kappa")
            kappa = float(dist_params["kappa"])
            theta = _nef_theta(dist_params)
            sqrt_t = math.sqrt(t)
            mu, scale, theta = nef_ghs_params(kappa, theta, sigma, sqrt_t, r, q, t)
            return nef_ghs_mgf(power, kappa, theta, mu, scale)
        if dist_name == "laplace":
            sqrt_t = math.sqrt(t)
            mu, scale = laplace_params(sigma, sqrt_t, r, q, t)
            denom = 1.0 - (scale * power) * (scale * power)
            if denom <= 0.0:
                raise ValueError("laplace requires |power| < 1/scale for finite MGF")
            return math.exp(mu * power) / denom
        if dist_name == "cgmy":
            if not dist_params:
                raise ValueError("cgmy requires dist_params")
            C_val = float(dist_params["C"])
            G_val = float(dist_params["G"])
            M_val = float(dist_params["M"])
            Y_val = float(dist_params["Y"])
            sqrt_t = math.sqrt(t)
            params_cgmy = cgmy_params(C_val, G_val, M_val, Y_val, sigma, sqrt_t, r, q, t)
            return cgmy_mgf(power, params_cgmy)

        sqrt_t = math.sqrt(t)
        pdf = _log_pdf_factory(dist_name, dist_params, sigma, sqrt_t, r, q, t, quad_limit)
        integrand = lambda x: math.exp(power * x) * pdf(x)
        edges = _integration_edges(sigma * sqrt_t)
        per_segment_limit = max(50, quad_limit // max(len(edges) - 1, 1))
        total = 0.0
        for start, end in zip(edges, edges[1:]):
            if end <= start:
                continue
            seg_val, _ = quad(
                integrand,
                start,
                end,
                limit=per_segment_limit,
                epsabs=1e-7,
                epsrel=1e-7,
            )
            total += seg_val
        return total

    e1 = moment_expect(1.0)
    e2 = moment_expect(2.0)
    e3 = moment_expect(3.0)
    e4 = moment_expect(4.0)
    s0_pow = s0
    s0_sq = s0 * s0
    e_st = s0_pow * e1
    e_st2 = s0_sq * e2
    e_st3 = (s0_sq * s0) * e3
    e_st4 = (s0_sq * s0_sq) * e4
    var = e_st2 - e_st * e_st
    std = math.sqrt(max(var, 0.0))
    if std == 0.0:
        return e_st, 0.0, 0.0, 3.0
    skew = (e_st3 - 3 * e_st * e_st2 + 2 * e_st ** 3) / (std ** 3)
    kurt = (e_st4 - 4 * e_st * e_st3 + 6 * e_st * e_st * e_st2 - 3 * e_st ** 4) / (var * var)
    return e_st, std, skew, kurt

    sqrt_t = math.sqrt(t)
    pdf = _log_pdf_factory(dist_name, dist_params, sigma, sqrt_t, r, q, t, quad_limit)

    def moment(n: int) -> float:
        integrand = lambda x: (x ** n) * pdf(x)
        val, _ = quad(integrand, -LOG_XMAX, LOG_XMAX, limit=quad_limit, epsabs=1e-7, epsrel=1e-7)
        return val

    m1 = moment(1)
    m2 = moment(2)
    m3 = moment(3)
    m4 = moment(4)
    var = m2 - m1 * m1
    mean_val = m1
    std_val = math.sqrt(max(var, 0.0))
    if std_val == 0.0:
        return mean_val, 0.0, 0.0, 3.0
    skew = (m3 - 3 * m2 * m1 + 2 * m1 ** 3) / (std_val ** 3)
    kurt = (m4 - 4 * m3 * m1 + 6 * m2 * m1 * m1 - 3 * m1 ** 4) / (var * var)
    return mean_val, std_val, skew, kurt


def option_price(
    cp: Literal["c", "p", "straddle"],
    s0: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    dist: Literal[
        "normal",
        "hypsecant",
        "logistic",
        "laplace",
        "tukeylambda",
        "johnsonsu",
        "hyperbolic",
        "champernowne",
        "ged",
        "nig",
        "vg",
        "gh",
        "cgmy",
        "nefghs",
        "gsh",
        "sgsh",
    ] = "normal",
    dist_params: Optional[Mapping[str, float]] = None,
    quad_limit: int = 200,
) -> float:
    """Price an option under the requested distribution."""
    params = dict(dist_params or {})
    if cp not in ("c", "p", "straddle"):
        raise ValueError('cp must be "c", "p", or "straddle"')
    if s0 <= 0.0:
        raise ValueError("s0 must be > 0")
    if k <= 0.0:
        raise ValueError("k must be > 0")
    if t < 0.0:
        raise ValueError("t must be >= 0")
    if sigma < 0.0:
        raise ValueError("sigma must be >= 0")

    if t == 0.0:
        if cp == "c":
            return max(s0 - k, 0.0)
        if cp == "p":
            return max(k - s0, 0.0)
        return abs(s0 - k)

    sqrt_t = math.sqrt(t)
    disc_r = math.exp(-r * t)
    disc_q = math.exp(-q * t)

    def call_normal() -> float:
        """Return call price for the normal distribution."""
        mu, vol_sqrt_t = normal_params(sigma, t, r, q)
        if sigma == 0.0 or vol_sqrt_t == 0.0:
            st = s0 * math.exp((r - q) * t)
            return disc_r * max(st - k, 0.0)
        d1 = (math.log(s0 / k) + (r - q + 0.5 * sigma * sigma) * t) / vol_sqrt_t
        d2 = d1 - vol_sqrt_t
        return s0 * disc_q * norm.cdf(d1) - k * disc_r * norm.cdf(d2)

    def call_hypsecant() -> float:
        """Return call price for the hyperbolic secant distribution."""
        mu, b = hypsecant_params(sigma, sqrt_t, r, q, t)
        if b == 0.0:
            st = s0 * math.exp((r - q) * t)
            return disc_r * max(st - k, 0.0)

        a = math.log(k / s0)
        y0 = (a - mu) / b

        def cdf_y(y: float) -> float:
            """Hypsecant CDF helper for tail probability."""
            z = 0.5 * math.pi * y
            if y >= 0.0:
                return 1.0 - (2.0 / math.pi) * math.atan(math.exp(-z))
            return (2.0 / math.pi) * math.atan(math.exp(z))

        p_tail = 1.0 - cdf_y(y0)

        def integrand(y: float) -> float:
            """Integrand for the tilted hypsecant expectation."""
            zabs = 0.5 * math.pi * abs(y)
            u = math.exp(-zabs)
            den = 1.0 + u * u
            return math.exp(b * y - zabs) / den

        j, _ = quad(integrand, y0, math.inf, limit=quad_limit)
        return disc_r * (s0 * math.exp(mu) * j - k * p_tail)

    def call_logistic() -> float:
        """Return call price for the logistic distribution."""
        if sigma == 0.0:
            st = s0 * math.exp((r - q) * t)
            return disc_r * max(st - k, 0.0)

        mu, scale = logistic_params(sigma, sqrt_t, r, q, t)
        a = math.log(k / s0)
        z = (a - mu) / scale
        p_tail = logistic_tail_prob(z)

        alpha = 1.0 - scale
        beta_param = 1.0 + scale
        tilted = math.exp(mu) * beta_fn(alpha, beta_param) * betainc(alpha, beta_param, p_tail)
        return disc_r * (s0 * tilted - k * p_tail)

    def call_laplace() -> float:
        """Return call price for the Laplace distribution."""
        if sigma == 0.0:
            st = s0 * math.exp((r - q) * t)
            return disc_r * max(st - k, 0.0)

        mu, scale = laplace_params(sigma, sqrt_t, r, q, t)
        a = math.log(k / s0)
        y0 = a - mu
        tail_prob = laplace_tail_prob(y0, scale)
        tilted = math.exp(mu) * laplace_tilted_tail(y0, scale)
        return disc_r * (s0 * tilted - k * tail_prob)

    def call_johnsonsu() -> float:
        """Return call price for the Johnson SU distribution."""
        if sigma == 0.0:
            st = s0 * math.exp((r - q) * t)
            return disc_r * max(st - k, 0.0)
        if "a" not in params or "b" not in params:
            raise ValueError('johnsonsu requires dist_params with keys "a" and "b"')
        a = float(params["a"])
        b = float(params["b"])
        loc, scale = johnson_su_params(a, b, sigma, sqrt_t, r, q, t, quad_limit)
        a_log = math.log(k / s0)
        tail_prob = johnson_su_tail_prob(a_log, a, b, loc, scale)
        tilted = johnson_su_tilted_tail(a_log, a, b, loc, scale, quad_limit)
        return disc_r * (s0 * tilted - k * tail_prob)

    def call_hyperbolic() -> float:
        """Return call price for the hyperbolic distribution."""
        if sigma == 0.0:
            st = s0 * math.exp((r - q) * t)
            return disc_r * max(st - k, 0.0)

        if "alpha" not in params or "beta" not in params:
            raise ValueError('hyperbolic requires dist_params with keys "alpha" and "beta"')
        alpha = float(params["alpha"])
        beta = float(params["beta"])
        mu, delta, alpha_eff, beta_eff = hyperbolic_params(alpha, beta, sigma, r, q, t)

        a = math.log(k / s0)
        tail_prob = hyperbolic_tail_prob(a, alpha_eff, beta_eff, delta, mu, quad_limit)
        tilted = hyperbolic_tilted_tail(a, alpha_eff, beta_eff, delta, mu, quad_limit)
        return disc_r * (s0 * tilted - k * tail_prob)
    def call_gh() -> float:
        """Return call price for the generalized hyperbolic distribution."""
        if sigma == 0.0:
            st = s0 * math.exp((r - q) * t)
            return disc_r * max(st - k, 0.0)
        if "lam" not in params or "alpha" not in params or "beta" not in params:
            raise ValueError('gh requires dist_params with keys "lam", "alpha", "beta"')
        lam = float(params["lam"])
        alpha = float(params["alpha"])
        beta = float(params["beta"])
        sqrt_t = math.sqrt(t)
        loc, scale = gh_params(lam, alpha, beta, sigma, sqrt_t, r, q, t, quad_limit)
        a = math.log(k / s0)
        tail_prob = gh_tail_prob(a, lam, alpha, beta, loc, scale)
        tilted = gh_tilted_tail(a, lam, alpha, beta, loc, scale, quad_limit)
        return disc_r * (s0 * tilted - k * tail_prob)

    def call_tukeylambda() -> float:
        """Return call price for the Tukey lambda distribution."""
        if sigma == 0.0:
            st = s0 * math.exp((r - q) * t)
            return disc_r * max(st - k, 0.0)
        if "lam" not in params:
            raise ValueError('tukeylambda requires dist_params with key "lam"')
        lam = float(params["lam"])
        loc, scale = tukey_lambda_params(lam, sigma, sqrt_t, r, q, t, quad_limit)
        a = math.log(k / s0)
        tail_prob = tukey_lambda_tail_prob(a, lam, loc, scale)
        tilted = tukey_lambda_tilted_tail(a, lam, loc, scale, quad_limit)
        return disc_r * (s0 * tilted - k * tail_prob)

    def call_champernowne() -> float:
        """Return call price for the Champernowne distribution."""
        if sigma == 0.0:
            st = s0 * math.exp((r - q) * t)
            return disc_r * max(st - k, 0.0)

        if "d" not in params:
            raise ValueError('champernowne requires dist_params with key "d"')
        d = float(params["d"])
        mu, scale, norm_const = champernowne_params(d, sigma, sqrt_t, r, q, t, quad_limit)

        a = math.log(k / s0)
        z0 = (a - mu) / scale
        tail_prob = champernowne_tail(z0, d, norm_const, quad_limit)
        tilted = math.exp(mu) * champernowne_tilted_tail(z0, d, norm_const, scale, quad_limit)
        return disc_r * (s0 * tilted - k * tail_prob)

    def call_ged() -> float:
        """Return call price for the generalized error distribution."""
        if sigma == 0.0:
            st = s0 * math.exp((r - q) * t)
            return disc_r * max(st - k, 0.0)

        if "p" not in params:
            raise ValueError('ged requires dist_params with key "p"')
        power = float(params["p"])
        mu, scale, norm_const = ged_params(power, sigma, sqrt_t, r, q, t, quad_limit)

        a = math.log(k / s0)
        y0 = a - mu
        tail_prob = ged_tail_prob(y0, scale, power, norm_const, quad_limit)
        tilted = math.exp(mu) * ged_tilted_tail(y0, scale, power, norm_const, quad_limit)
        return disc_r * (s0 * tilted - k * tail_prob)

    def call_nig() -> float:
        """Return call price for the normal-inverse-Gaussian distribution."""
        if sigma == 0.0:
            st = s0 * math.exp((r - q) * t)
            return disc_r * max(st - k, 0.0)

        if "alpha" not in params or "beta" not in params:
            raise ValueError('nig requires dist_params with keys "alpha" and "beta"')
        alpha = float(params["alpha"])
        beta = float(params["beta"])
        if alpha <= 0.0:
            raise ValueError("nig alpha must be > 0")
        if abs(beta) >= alpha:
            raise ValueError("nig requires |beta| < alpha")

        variance = sigma * sigma * t
        delta = nig_delta(variance, alpha, beta)
        if delta <= 0.0:
            raise ValueError("nig delta must be > 0")
        if abs(beta + 1.0) >= alpha:
            raise ValueError("nig requires |beta + 1| < alpha so E[exp(X)] is finite")

        log_mgf = delta * (
            math.sqrt(alpha * alpha - beta * beta)
            - math.sqrt(alpha * alpha - (beta + 1.0) ** 2)
        )
        mu = (r - q) * t - log_mgf

        a = math.log(k / s0)
        tail_prob = nig_tail_prob(a, alpha, beta, delta, mu, quad_limit)
        tilted = nig_tilted_tail(a, alpha, beta, delta, mu, quad_limit)
        return disc_r * (s0 * tilted - k * tail_prob)

    def call_vg() -> float:
        """Return call price for the variance-gamma distribution."""
        if sigma == 0.0:
            st = s0 * math.exp((r - q) * t)
            return disc_r * max(st - k, 0.0)

        if "theta" not in params or "nu" not in params:
            raise ValueError('vg requires dist_params with keys "theta" and "nu"')
        theta = float(params["theta"])
        nu = float(params["nu"])
        if nu <= 0.0:
            raise ValueError("vg parameter nu must be > 0")

        sigma_sq_target = sigma * sigma
        sigma_b_sq = sigma_sq_target - nu * theta * theta
        if sigma_b_sq <= 0.0:
            raise ValueError("vg requires sigma^2 > nu * theta^2 to match variance")
        sigma_b = math.sqrt(sigma_b_sq)

        arg = 1.0 - nu * (theta + 0.5 * sigma_b_sq)
        if arg <= 0.0:
            raise ValueError("vg requires 1 - nu*(theta + 0.5*sigma_b^2) > 0 for finite mgf")

        mu_adj = (r - q) * t + (t / nu) * math.log(arg)
        shape = t / nu
        scale = nu
        undisc = vg_call_expectation(s0, k, mu_adj, theta, sigma_b, shape, scale, quad_limit)
        return disc_r * undisc

    def call_from_pdf(dist_name_local: str, params_local: Mapping[str, float]) -> float:
        """Generic numerical integration for distributions with pdf only."""
        pdf = _log_pdf_factory(
            dist_name_local,
            params_local,
            sigma,
            sqrt_t,
            r,
            q,
            t,
            quad_limit,
        )
        lower = math.log(k / s0)

        def tail_integrand(x: float) -> float:
            return pdf(x)

        def tilted_integrand(x: float) -> float:
            return math.exp(x) * pdf(x)

        tail, _ = quad(
            tail_integrand,
            lower,
            LOG_XMAX,
            limit=quad_limit,
            epsabs=1e-7,
            epsrel=1e-7,
        )
        tail = min(max(tail, 0.0), 1.0)
        tilted, _ = quad(
            tilted_integrand,
            lower,
            LOG_XMAX,
            limit=quad_limit,
            epsabs=1e-7,
            epsrel=1e-7,
        )
        tilted = max(tilted, 0.0)
        return disc_r * (s0 * tilted - k * tail)

    def call_cgmy() -> float:
        """Return call price for CGMY using a damped Fourier integral."""
        if sigma == 0.0:
            st = s0 * math.exp((r - q) * t)
            return disc_r * max(st - k, 0.0)
        if not {"C", "G", "M", "Y"} <= params.keys():
            raise ValueError('cgmy requires dist_params with keys "C", "G", "M", and "Y"')
        C_val = float(params["C"])
        G_val = float(params["G"])
        M_val = float(params["M"])
        Y_val = float(params["Y"])
        sqrt_t = math.sqrt(t)
        params_cgmy = cgmy_params(C_val, G_val, M_val, Y_val, sigma, sqrt_t, r, q, t)
        alpha_damp = 1.5
        shift = alpha_damp + 1.0
        scale = params_cgmy.scale
        mgf_bound = min(G_val, M_val)
        if scale * shift >= mgf_bound:
            raise ValueError("cgmy damping requires scale*(alpha+1) < min(G,M)")
        log_k = math.log(k)
        log_s0 = math.log(s0)

        def phi_log_st(u: complex) -> complex:
            """Characteristic function of log S_T for CGMY."""
            return cmath.exp(1j * u * log_s0) * cgmy_characteristic(u, params_cgmy)

        def integrand(u: float) -> float:
            z = u - 1j * shift
            phi_val = phi_log_st(z)
            numerator = cmath.exp(-1j * u * log_k) * phi_val
            denom = (alpha_damp * alpha_damp + alpha_damp - u * u) + 1j * u * (2.0 * alpha_damp + 1.0)
            return (numerator / denom).real

        u_max = CGMY_U_MAX / max(math.sqrt(t), 1e-6)
        integral, _ = quad(integrand, 0.0, u_max, limit=quad_limit, epsabs=1e-9, epsrel=1e-9)
        price = math.exp(-alpha_damp * log_k) / math.pi * integral
        return disc_r * price

    if dist == "normal":
        c = call_normal()
    elif dist == "hypsecant":
        c = call_hypsecant()
    elif dist == "logistic":
        c = call_logistic()
    elif dist == "laplace":
        c = call_laplace()
    elif dist == "tukeylambda":
        c = call_tukeylambda()
    elif dist == "johnsonsu":
        c = call_johnsonsu()
    elif dist == "hyperbolic":
        c = call_hyperbolic()
    elif dist == "champernowne":
        c = call_champernowne()
    elif dist == "ged":
        c = call_ged()
    elif dist == "nig":
        c = call_nig()
    elif dist == "vg":
        c = call_vg()
    elif dist == "gh":
        c = call_gh()
    elif dist == "nefghs":
        c = call_from_pdf("nefghs", params)
    elif dist == "cgmy":
        c = call_cgmy()
    elif dist == "gsh":
        c = call_from_pdf("gsh", params)
    elif dist == "sgsh":
        c = call_from_pdf("sgsh", params)
    else:
        raise ValueError(
            'dist must be "normal", "hypsecant", "logistic", "laplace", "tukeylambda", "johnsonsu", "hyperbolic", '
            '"champernowne", "ged", "nig", "vg", "gh", "cgmy", "nefghs", "gsh", or "sgsh"'
        )

    p = c - s0 * disc_q + k * disc_r

    if cp == "c":
        return c
    if cp == "p":
        return p
    return c + p
    

