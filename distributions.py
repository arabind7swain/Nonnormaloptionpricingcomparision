import math
import cmath
from functools import lru_cache
from typing import NamedTuple, Tuple

from scipy.integrate import quad
from scipy.special import beta as beta_fn, betainc, kv, gammaln, loggamma
from scipy.stats import norm, johnsonsu, tukeylambda, genhyperbolic

CHAMPER_ZMAX = 40.0
GED_ZMAX = 40.0
ALLOWED_GED_POWERS = (1.0, 1.5, 2.0)
NIG_XMAX = 40.0
HYPERBOLIC_XMAX = 40.0
JOHNSON_ZMAX = 12.0
TUKEY_LIMIT = 40.0
NEF_GHS_XMAX = 40.0
NEF_GHS_MGF_LIMIT = 200
GSH_XMAX = 50.0
GH_ZMAX = 50.0
# parameters for CGMY Fourier integrals
CGMY_U_MAX = 200.0
# parameters for VG pdf approximation (truncate series)
VG_G_SCALE_FACTOR = 10.0
VG_G_MIN = 1e-16
VG_PDF_GTERMS = 40
_GSH_CONST_CACHE: dict[tuple[float, int], tuple[float, float, float]] = {}
_GSH_MGF_CACHE: dict[tuple[float, float, int], float] = {}
_GSH_ABS_CACHE: dict[tuple[float, int], float] = {}
_SGSH_MGF_CACHE: dict[tuple[float, float, float, int], float] = {}
_SGSH_MOMENT_CACHE: dict[tuple[float, float, int], tuple[float, float]] = {}
_SGSH_HIGHER_CACHE: dict[tuple[float, float, int], tuple[float, float]] = {}
_NEF_GHS_PSI_CACHE: dict[tuple[float, float, int], float] = {}
_NEF_GHS_MOM_CACHE: dict[tuple[float, float, int], Tuple[float, float, float, float]] = {}


class CgmyParams(NamedTuple):
    """Store calibrated CGMY parameters for reuse."""

    C: float
    G: float
    M: float
    Y: float
    scale: float
    mu: float
    gamma_neg: float
    t: float


def quad_limit_value(limit: int) -> int:
    """Normalize the quad integration limit for numerical routines."""
    return max(50, int(limit))


def logistic_tail_prob(z: float) -> float:
    """Return P(Y>z) for a standard logistic variable."""
    if z >= 0.0:
        ez = math.exp(-z)
        return ez / (1.0 + ez)
    ez = math.exp(z)
    return 1.0 / (1.0 + ez)


def normal_params(sigma: float, t: float, r: float, q: float) -> Tuple[float, float]:
    """Return (mu, vol_sqrt_t) for the normal log-return distribution."""
    mu = (r - q - 0.5 * sigma * sigma) * t
    vol = sigma * math.sqrt(t)
    return mu, vol


def hypsecant_params(sigma: float, sqrt_t: float, r: float, q: float, t: float) -> Tuple[float, float]:
    """Return (mu, scale) for the hyperbolic secant case."""
    b = sigma * sqrt_t
    if b >= 0.5 * math.pi:
        raise ValueError("hypsecant requires sigma*sqrt(t) < pi/2 so E[exp(X)] is finite")
    mu = (r - q) * t + math.log(math.cos(b))
    return mu, b


def logistic_params(sigma: float, sqrt_t: float, r: float, q: float, t: float) -> Tuple[float, float]:
    """Return (mu, scale) for the logistic case."""
    scale = sigma * sqrt_t * math.sqrt(3.0) / math.pi
    if scale >= 1.0:
        raise ValueError("logistic requires scale = sigma*sqrt(t)*sqrt(3)/pi < 1")
    mgf = (math.pi * scale) / math.sin(math.pi * scale)
    mu = (r - q) * t - math.log(mgf)
    return mu, scale


def _stable_sech(z: float) -> float:
    """Evaluate sech(z) without cosh overflow in the far tails."""
    az = abs(z)
    if az < 30.0:
        return 1.0 / math.cosh(z)
    # For large |z|, cosh(z) ≈ 0.5 * exp(|z|), so sech(z) ≈ 2 * exp(-|z|).
    return 2.0 * math.exp(-az)


def hypsecant_pdf_val(x: float, mu: float, scale: float) -> float:
    """Hyperbolic secant pdf evaluated at x."""
    if scale <= 0.0:
        return 0.0
    z = (x - mu) / scale
    return 0.5 / scale * _stable_sech(0.5 * math.pi * z)


def logistic_pdf_val(x: float, mu: float, scale: float) -> float:
    """Logistic pdf evaluated at x."""
    if scale <= 0.0:
        return 0.0
    z = (x - mu) / scale
    if z >= 0.0:
        ez = math.exp(-z)
        denom = 1.0 + ez
        return ez / (scale * denom * denom)
    ez = math.exp(z)
    denom = 1.0 + ez
    return ez / (scale * denom * denom)


def laplace_params(sigma: float, sqrt_t: float, r: float, q: float, t: float) -> Tuple[float, float]:
    """Return (mu, scale) for the Laplace case."""
    scale = sigma * sqrt_t / math.sqrt(2.0)
    if scale >= 1.0:
        raise ValueError("laplace requires sigma*sqrt(t)/sqrt(2) < 1 so E[exp(X)] is finite")
    mu = (r - q) * t + math.log(1.0 - scale * scale)
    return mu, scale


def laplace_tail_prob(y0: float, scale: float) -> float:
    """Tail probability for Laplace(0, scale)."""
    if y0 >= 0.0:
        return 0.5 * math.exp(-y0 / scale)
    return 1.0 - 0.5 * math.exp(y0 / scale)


def laplace_tilted_tail(y0: float, scale: float) -> float:
    """Return E[exp(Y) 1_{Y>y0}] for Laplace(0, scale)."""
    pref = 0.5 / scale
    if y0 >= 0.0:
        denom = (1.0 / scale) - 1.0
        if denom <= 0.0:
            raise ValueError("Laplace tilt requires scale < 1")
        return pref * math.exp(y0 * (1.0 - 1.0 / scale)) / denom
    pos_denom = (1.0 / scale) - 1.0
    if pos_denom <= 0.0:
        raise ValueError("Laplace tilt requires scale < 1")
    neg_denom = 1.0 + (1.0 / scale)
    neg_term = pref * (1.0 - math.exp(y0 * (1.0 + 1.0 / scale))) / neg_denom
    pos_term = pref / pos_denom
    return neg_term + pos_term


def champernowne_h(z: float, d: float) -> float:
    """Evaluate the Champernowne base numerator h_d(z)."""
    abs_z = abs(z)
    if abs_z < 40.0:
        return 1.0 / (2.0 * math.cosh(z) + d)
    ez = math.exp(-abs_z)
    # Rewrite denominator to avoid cosh overflow: 2*cosh(z)+d = e^{|z|}(1+e^{-2|z|}) + d
    denom = 1.0 + ez * ez + d * ez
    return ez / denom


@lru_cache(maxsize=None)
def champernowne_base_stats(d: float, limit: int) -> Tuple[float, float]:
    """Return normalization constant and second moment for Champernowne(d)."""
    lim = quad_limit_value(limit)

    def h(z: float) -> float:
        """Champernowne integrand for normalization."""
        return champernowne_h(z, d)

    norm = 2.0 * quad(h, 0.0, CHAMPER_ZMAX, limit=lim, epsabs=1e-12, epsrel=1e-12)[0]
    second = 2.0 * quad(
        lambda z: z * z * h(z),
        0.0,
        CHAMPER_ZMAX,
        limit=lim,
        epsabs=1e-12,
        epsrel=1e-12,
    )[0] / norm
    return norm, second


def champernowne_pdf(z: float, d: float, norm_const: float) -> float:
    """Champernowne(d) pdf at z with normalization constant."""
    return champernowne_h(z, d) / norm_const


def champernowne_tail(z0: float, d: float, norm_const: float, limit: int) -> float:
    """Tail probability P(Z>z0) for Champernowne(d)."""
    lim = quad_limit_value(limit)
    upper = CHAMPER_ZMAX
    if z0 <= -CHAMPER_ZMAX:
        return 1.0
    if z0 >= CHAMPER_ZMAX:
        return 0.0
    lower = max(z0, -CHAMPER_ZMAX)
    val, _ = quad(
        lambda z: champernowne_pdf(z, d, norm_const),
        lower,
        upper,
        limit=lim,
        epsabs=1e-12,
        epsrel=1e-12,
    )
    return val


def champernowne_tilted_tail(
    z0: float,
    d: float,
    norm_const: float,
    scale: float,
    limit: int,
) -> float:
    """Return E[exp(scale*Z) 1_{Z>z0}] for Champernowne(d)."""
    lim = quad_limit_value(limit)

    def pdf(z: float) -> float:
        """Champernowne(d) pdf for tilt integration."""
        return champernowne_pdf(z, d, norm_const)

    if z0 <= -CHAMPER_ZMAX:
        pos, _ = quad(
            lambda z: math.exp(scale * z) * pdf(z),
            -CHAMPER_ZMAX,
            CHAMPER_ZMAX,
            limit=lim,
            epsabs=1e-12,
            epsrel=1e-12,
        )
        return pos
    if z0 >= CHAMPER_ZMAX:
        return 0.0
    lower = max(z0, -CHAMPER_ZMAX)
    val, _ = quad(
        lambda z: math.exp(scale * z) * pdf(z),
        lower,
        CHAMPER_ZMAX,
        limit=lim,
        epsabs=1e-12,
        epsrel=1e-12,
    )
    return val


def champernowne_mgf(
    d: float,
    scale: float,
    norm_const: float,
    limit: int,
    u: float = 1.0,
) -> float:
    """Moment generating function E[exp(u*scale*Z)] for Champernowne(d)."""
    if abs(scale * u) >= 1.0:
        raise ValueError("champernowne requires |scale| < 1 for finite mgf")
    lim = quad_limit_value(limit)
    val, _ = quad(
        lambda z: math.cosh(scale * u * z) * champernowne_h(z, d),
        0.0,
        CHAMPER_ZMAX,
        limit=lim,
        epsabs=1e-12,
        epsrel=1e-12,
    )
    return 2.0 * val / norm_const


def champernowne_params(
    d: float,
    sigma: float,
    sqrt_t: float,
    r: float,
    q: float,
    t: float,
    quad_limit: int,
) -> Tuple[float, float, float]:
    """Return (mu, scale, norm_const) for Champernowne(d)."""
    if d <= -2.0:
        raise ValueError("champernowne shape parameter d must be > -2")
    norm_const, ez2 = champernowne_base_stats(d, quad_limit)
    sd_z = math.sqrt(ez2)
    if sd_z <= 0.0:
        raise ValueError("invalid Champernowne variance")
    scale = sigma * sqrt_t / sd_z
    if scale >= 1.0:
        raise ValueError("champernowne requires sigma*sqrt(t)/sd_Z < 1 so E[exp(X)] is finite")
    mgf = champernowne_mgf(d, scale, norm_const, quad_limit)
    mu = (r - q) * t - math.log(mgf)
    return mu, scale, norm_const


@lru_cache(maxsize=None)
def ged_gamma_values(power: float) -> Tuple[float, float]:
    """Return gamma(1/p) and gamma(3/p) for GED power p."""
    gamma1 = math.gamma(1.0 / power)
    gamma3 = math.gamma(3.0 / power)
    return gamma1, gamma3


def ged_norm_const(scale: float, power: float) -> float:
    """Normalization constant for GED(scale, power)."""
    gamma1, _ = ged_gamma_values(power)
    return power / (2.0 * scale * gamma1)


def ged_tail_prob(
    y0: float,
    scale: float,
    power: float,
    norm_const: float,
    limit: int,
) -> float:
    """Tail probability P(Y>y0) for GED."""
    lim = quad_limit_value(limit)
    upper = GED_ZMAX
    if y0 >= GED_ZMAX:
        return 0.0
    if y0 <= -GED_ZMAX:
        return 1.0
    lower = max(y0, -GED_ZMAX)
    integrand = lambda y: norm_const * math.exp(-((abs(y) / scale) ** power))
    val, _ = quad(integrand, lower, upper, limit=lim, epsabs=1e-12, epsrel=1e-12)
    return val


def ged_tilted_tail(
    y0: float,
    scale: float,
    power: float,
    norm_const: float,
    limit: int,
) -> float:
    """Return E[exp(Y) 1_{Y>y0}] for GED."""
    lim = quad_limit_value(limit)
    upper = GED_ZMAX
    if y0 >= GED_ZMAX:
        return 0.0
    lower = max(y0, -GED_ZMAX)
    integrand = lambda y: math.exp(y) * norm_const * math.exp(-((abs(y) / scale) ** power))
    val, _ = quad(integrand, lower, upper, limit=lim, epsabs=1e-12, epsrel=1e-12)
    return val


def ged_mgf(
    scale: float,
    power: float,
    norm_const: float,
    limit: int,
    u: float = 1.0,
) -> float:
    """Moment generating function E[exp(uY)] for GED."""
    lim = quad_limit_value(limit)
    integrand = lambda y: math.cosh(u * y) * norm_const * math.exp(-((y / scale) ** power))
    val, _ = quad(integrand, 0.0, GED_ZMAX, limit=lim, epsabs=1e-12, epsrel=1e-12)
    return 2.0 * val


def ged_params(
    power: float,
    sigma: float,
    sqrt_t: float,
    r: float,
    q: float,
    t: float,
    quad_limit: int,
) -> Tuple[float, float, float]:
    """Return (mu, scale, norm_const) for the GED with the given power."""
    if power not in ALLOWED_GED_POWERS:
        raise ValueError(f"ged power must be one of {ALLOWED_GED_POWERS}")
    gamma1, gamma3 = ged_gamma_values(power)
    if gamma1 <= 0.0 or gamma3 <= 0.0:
        raise ValueError("invalid GED gamma values")
    scale = sigma * sqrt_t / math.sqrt(gamma3 / gamma1)
    if power == 1.0 and scale >= 1.0:
        raise ValueError("ged with p=1 requires sigma*sqrt(t)/sqrt(2) < 1")
    norm_const = ged_norm_const(scale, power)
    mgf = ged_mgf(scale, power, norm_const, quad_limit)
    mu = (r - q) * t - math.log(mgf)
    return mu, scale, norm_const


@lru_cache(maxsize=None)
def johnson_su_base_stats(a: float, b: float) -> Tuple[float, float]:
    """Return (mean, std) for Johnson SU(a,b) with loc=0, scale=1."""
    if b <= 0.0:
        raise ValueError("johnsonsu requires b>0")
    mean, var = johnsonsu.stats(a, b, moments="mv")
    if var <= 0.0:
        raise ValueError("invalid johnsonsu variance")
    return float(mean), math.sqrt(float(var))


def johnson_su_mgf(scale: float, a: float, b: float, limit: int) -> float:
    """Return E[exp(scale*S)] for standard Johnson SU S."""
    lim = quad_limit_value(limit)

    def integrand(z: float) -> float:
        """Integrand over the auxiliary normal variable."""
        s = math.sinh((z - a) / b)
        return math.exp(scale * s) * norm.pdf(z)

    val, _ = quad(
        integrand,
        -JOHNSON_ZMAX,
        JOHNSON_ZMAX,
        limit=lim,
        epsabs=1e-8,
        epsrel=1e-8,
    )
    return val


@lru_cache(maxsize=None)
def johnson_su_params(
    a: float,
    b: float,
    sigma: float,
    sqrt_t: float,
    r: float,
    q: float,
    t: float,
    quad_limit: int,
) -> Tuple[float, float]:
    """Return (loc, scale) for Johnson SU log returns."""
    if b <= 0.0:
        raise ValueError("johnsonsu requires b>0")
    _, std_base = johnson_su_base_stats(a, b)
    if std_base <= 0.0:
        raise ValueError("invalid johnsonsu base std")
    scale = sigma * sqrt_t / std_base
    if scale <= 0.0:
        raise ValueError("johnsonsu scale must be positive")
    mgf1 = johnson_su_mgf(scale, a, b, quad_limit)
    if mgf1 <= 0.0 or not math.isfinite(mgf1):
        raise ValueError("johnsonsu mgf evaluation failed")
    loc = (r - q) * t - math.log(mgf1)
    return loc, scale


def johnson_su_pdf(x: float, a: float, b: float, loc: float, scale: float) -> float:
    """Johnson SU pdf evaluated at x."""
    return johnsonsu.pdf(x, a, b, loc=loc, scale=scale)


def johnson_su_tail_prob(
    x0: float,
    a: float,
    b: float,
    loc: float,
    scale: float,
) -> float:
    """Tail probability P(X>x0) for Johnson SU."""
    z = (x0 - loc) / scale
    return johnsonsu.sf(z, a, b)


def johnson_su_tilted_tail(
    x0: float,
    a: float,
    b: float,
    loc: float,
    scale: float,
    limit: int,
) -> float:
    """Return E[exp(X) 1_{X>x0}] for Johnson SU."""
    if scale <= 0.0:
        raise ValueError("johnsonsu scale must be positive")
    lim = quad_limit_value(limit)
    threshold = (x0 - loc) / scale
    z_thresh = a + b * math.asinh(threshold)
    if z_thresh >= JOHNSON_ZMAX:
        return 0.0
    lower = max(z_thresh, -JOHNSON_ZMAX)

    def integrand(z: float) -> float:
        """Integrand over the auxiliary normal variable."""
        s = math.sinh((z - a) / b)
        return math.exp(loc + scale * s) * norm.pdf(z)

    val, _ = quad(
        integrand,
        lower,
        JOHNSON_ZMAX,
        limit=lim,
        epsabs=1e-8,
        epsrel=1e-8,
    )
    return val


@lru_cache(maxsize=None)
def tukey_lambda_moment_values(lam: float) -> Tuple[float, float, float, float]:
    """Return base mean/var/skew/kurtosis for Tukey lambda(lam)."""
    try:
        mean, var, skew, kurt_excess = tukeylambda.stats(lam, moments="mvsk")
    except Exception:  # pragma: no cover - SciPy may raise for exotic params.
        mean = var = skew = kurt_excess = math.nan
    if (
        math.isfinite(mean)
        and math.isfinite(var)
        and math.isfinite(skew)
        and math.isfinite(kurt_excess)
        and var > 0.0
    ):
        return float(mean), float(var), float(skew), float(kurt_excess + 3.0)

    # Numerical fallback if SciPy stats are unavailable.
    a_base, b_base = tukeylambda.support(lam)
    lim = quad_limit_value(200)

    def base_pdf(z: float) -> float:
        """Return pdf of Tukey lambda at z for integration."""
        return tukeylambda.pdf(z, lam)

    def raw_moment(power: int) -> float:
        """Return E[Z**power] for the base Tukey lambda variable Z."""
        val, _ = quad(
            lambda z: (z**power) * base_pdf(z),
            a_base,
            b_base,
            limit=lim,
            epsabs=1e-10,
            epsrel=1e-10,
        )
        return val

    m1 = raw_moment(1)
    m2 = raw_moment(2)
    variance = m2 - m1 * m1
    if variance <= 0.0:
        return m1, 0.0, 0.0, 3.0
    std = math.sqrt(variance)
    m3 = raw_moment(3)
    m4 = raw_moment(4)
    skew = (m3 - 3.0 * m2 * m1 + 2.0 * m1**3) / (std**3)
    kurt = (m4 - 4.0 * m3 * m1 + 6.0 * m2 * m1 * m1 - 3.0 * m1**4) / (variance * variance)
    return m1, variance, skew, kurt


@lru_cache(maxsize=None)
def tukey_lambda_stats(lam: float) -> Tuple[float, float]:
    """Return base mean and std for Tukey lambda(lam)."""
    mean, var, _, _ = tukey_lambda_moment_values(lam)
    if not math.isfinite(mean) or not math.isfinite(var) or var <= 0.0:
        raise ValueError("invalid tukey lambda stats; check lambda parameter")
    return float(mean), math.sqrt(float(var))


def tukey_lambda_mgf(scale: float, lam: float, limit: int) -> float:
    """Return E[exp(scale*S)] for S~TukeyLambda(lam, loc=0, scale=1)."""
    lim = quad_limit_value(limit)
    a, b = tukeylambda.support(lam)

    def integrand(x: float) -> float:
        return math.exp(scale * x) * tukeylambda.pdf(x, lam)

    val, _ = quad(
        integrand,
        a,
        b,
        limit=lim,
        epsabs=1e-8,
        epsrel=1e-8,
    )
    return val


@lru_cache(maxsize=None)
def tukey_lambda_params(
    lam: float,
    sigma: float,
    sqrt_t: float,
    r: float,
    q: float,
    t: float,
    quad_limit: int,
) -> Tuple[float, float]:
    """Return (loc, scale) for Tukey-lambda log returns."""
    if lam <= 0.0:
        raise ValueError("tukeylambda currently requires lambda > 0 for finite mgf")
    _, std_base = tukey_lambda_stats(lam)
    if std_base <= 0.0:
        raise ValueError("invalid base Tukey lambda std")
    scale = sigma * sqrt_t / std_base
    if scale <= 0.0:
        raise ValueError("tukeylambda scale must be positive")
    mgf1 = tukey_lambda_mgf(scale, lam, quad_limit)
    if mgf1 <= 0.0 or not math.isfinite(mgf1):
        raise ValueError("tukeylambda mgf evaluation failed")
    loc = (r - q) * t - math.log(mgf1)
    return loc, scale


def tukey_lambda_pdf(x: float, lam: float, loc: float, scale: float) -> float:
    """Tukey lambda pdf evaluated at x."""
    if scale <= 0.0:
        raise ValueError("tukeylambda scale must be positive")
    z = (x - loc) / scale
    return tukeylambda.pdf(z, lam) / scale


def tukey_lambda_tail_prob(x0: float, lam: float, loc: float, scale: float) -> float:
    """Tail probability P(X>x0) for Tukey lambda."""
    if scale <= 0.0:
        raise ValueError("tukeylambda scale must be positive")
    z = (x0 - loc) / scale
    return tukeylambda.sf(z, lam)


def tukey_lambda_tilted_tail(
    x0: float,
    lam: float,
    loc: float,
    scale: float,
    limit: int,
) -> float:
    """Return E[exp(X) 1_{X>x0}] for Tukey lambda."""
    if scale <= 0.0:
        raise ValueError("tukeylambda scale must be positive")
    lim = quad_limit_value(limit)
    s_lower = (x0 - loc) / scale
    a_base, b_base = tukeylambda.support(lam)
    lower = max(s_lower, a_base)
    upper = b_base
    if lower >= upper:
        return 0.0

    def integrand(s: float) -> float:
        return math.exp(loc + scale * s) * tukeylambda.pdf(s, lam)

    val, _ = quad(
        integrand,
        lower,
        upper,
        limit=lim,
        epsabs=1e-8,
        epsrel=1e-8,
    )
    return val


def _hyperbolic_variance(alpha: float, beta: float, delta: float) -> float:
    """Return the variance of Hyperbolic(alpha,beta,delta,mu=0)."""
    gamma = math.sqrt(alpha * alpha - beta * beta)
    if gamma <= 0.0:
        raise ValueError("hyperbolic requires alpha>|beta|")
    z = delta * gamma
    k1 = kv(1, z)
    k2 = kv(2, z)
    k3 = kv(3, z)
    if k1 <= 0.0 or math.isinf(k1):
        raise ValueError("invalid Bessel evaluation in hyperbolic variance")
    ratio = k2 / k1
    term1 = delta * ratio / gamma
    diff = k3 / k1 - ratio * ratio
    term2 = (delta * beta / gamma) ** 2 * diff
    return term1 + term2


def hyperbolic_mgf_unit(u: float, alpha: float, beta: float, delta: float) -> float:
    """Return mgf E[exp(uX)] for Hyperbolic(alpha,beta,delta,mu=0)."""
    gamma = math.sqrt(alpha * alpha - beta * beta)
    arg = alpha * alpha - (beta + u) * (beta + u)
    if arg <= 0.0:
        raise ValueError("hyperbolic requires |beta+u| < alpha for mgf")
    sqrt_arg = math.sqrt(arg)
    z0 = delta * gamma
    z1 = delta * sqrt_arg
    k1_z0 = kv(1, z0)
    k1_z1 = kv(1, z1)
    if k1_z0 <= 0.0 or k1_z1 <= 0.0:
        raise ValueError("invalid Bessel evaluation in hyperbolic mgf")
    return (gamma / sqrt_arg) * (k1_z1 / k1_z0)


@lru_cache(maxsize=None)
def hyperbolic_params(
    alpha: float,
    beta: float,
    sigma: float,
    r: float,
    q: float,
    t: float,
) -> Tuple[float, float, float, float]:
    """Return (mu, delta, alpha_eff, beta_eff) for hyperbolic log returns."""
    if alpha <= 0.0:
        raise ValueError("hyperbolic alpha must be > 0")
    if abs(beta) >= alpha:
        raise ValueError("hyperbolic requires |beta| < alpha")
    if t <= 0.0 or sigma <= 0.0:
        raise ValueError("hyperbolic requires sigma>0 and t>0")
    variance_target = sigma * sigma * t
    base_delta = 1.0
    base_var = _hyperbolic_variance(alpha, beta, base_delta)
    if base_var <= 0.0:
        raise ValueError("hyperbolic base variance must be positive")
    scale = math.sqrt(variance_target / base_var)
    if scale <= 0.0:
        raise ValueError("hyperbolic scaling factor must be positive")
    alpha_eff = alpha / scale
    beta_eff = beta / scale
    delta_eff = base_delta * scale
    if abs(beta_eff) >= alpha_eff:
        raise ValueError("hyperbolic scaling invalid; choose different alpha/beta")
    if abs(beta_eff + 1.0) >= alpha_eff:
        raise ValueError("hyperbolic requires |beta_eff+1| < alpha_eff to keep mgf finite at 1")
    mgf1 = hyperbolic_mgf_unit(1.0, alpha_eff, beta_eff, delta_eff)
    mu = (r - q) * t - math.log(mgf1)
    return mu, delta_eff, alpha_eff, beta_eff


def hyperbolic_pdf(x: float, alpha: float, beta: float, delta: float, mu: float) -> float:
    """Hyperbolic(alpha,beta,delta,mu) pdf."""
    gamma = math.sqrt(alpha * alpha - beta * beta)
    z = delta * gamma
    k1 = kv(1, z)
    if k1 <= 0.0:
        raise ValueError("invalid Bessel evaluation in hyperbolic pdf")
    norm = gamma / (2.0 * alpha * delta * k1)
    return norm * math.exp(-alpha * math.sqrt(delta * delta + (x - mu) ** 2) + beta * (x - mu))


def hyperbolic_tail_prob(
    x0: float,
    alpha: float,
    beta: float,
    delta: float,
    mu: float,
    limit: int,
) -> float:
    """Tail probability P(X>x0) for the hyperbolic distribution."""
    lim = quad_limit_value(limit)
    if x0 >= HYPERBOLIC_XMAX:
        return 0.0
    if x0 <= -HYPERBOLIC_XMAX:
        return 1.0
    lower = max(x0, -HYPERBOLIC_XMAX)
    val, _ = quad(
        lambda x: hyperbolic_pdf(x, alpha, beta, delta, mu),
        lower,
        HYPERBOLIC_XMAX,
        limit=lim,
        epsabs=1e-12,
        epsrel=1e-12,
    )
    return val


def hyperbolic_tilted_tail(
    x0: float,
    alpha: float,
    beta: float,
    delta: float,
    mu: float,
    limit: int,
) -> float:
    """Return E[exp(X) 1_{X>x0}] for the hyperbolic distribution."""
    lim = quad_limit_value(limit)
    if x0 >= HYPERBOLIC_XMAX:
        return 0.0
    lower = max(x0, -HYPERBOLIC_XMAX)
    val, _ = quad(
        lambda x: math.exp(x) * hyperbolic_pdf(x, alpha, beta, delta, mu),
        lower,
        HYPERBOLIC_XMAX,
        limit=lim,
        epsabs=1e-12,
        epsrel=1e-12,
    )
    return val


@lru_cache(maxsize=None)
def _ghs_log_const(kappa: float) -> float:
    """Return log of the normalization constant for GHS(kappa)."""
    if kappa <= 0.0:
        raise ValueError("ghs kappa must be > 0")
    return (2.0 * kappa - 2.0) * math.log(2.0) - math.log(math.pi) - gammaln(2.0 * kappa)


def _ghs_log_pdf(z: float, kappa: float) -> float:
    """Return log pdf of the base GHS distribution."""
    log_gamma = loggamma(kappa + 0.5j * z)
    return _ghs_log_const(kappa) + 2.0 * log_gamma.real


def _nef_ghs_log_partition(theta: float, kappa: float, limit: int) -> float:
    """Return log mgf of the base GHS distribution evaluated at theta."""
    if abs(theta) >= 1.0:
        raise ValueError("nef_ghs canonical theta must satisfy |theta| < 1")
    lim = quad_limit_value(limit)
    key = (round(theta, 12), round(kappa, 12), lim)
    cached = _NEF_GHS_PSI_CACHE.get(key)
    if cached is not None:
        return cached

    lower = -NEF_GHS_XMAX
    upper = NEF_GHS_XMAX

    def integrand(z: float) -> float:
        return math.exp(theta * z + _ghs_log_pdf(z, kappa))

    val, _ = quad(
        integrand,
        lower,
        upper,
        limit=lim,
        epsabs=1e-9,
        epsrel=1e-9,
    )
    if val <= 0.0 or not math.isfinite(val):
        raise ValueError("nef_ghs log-partition evaluation failed")
    psi_val = math.log(val)
    _NEF_GHS_PSI_CACHE[key] = psi_val
    return psi_val


def nef_ghs_base_moments(
    theta: float,
    kappa: float,
    limit: int,
) -> Tuple[float, float, float, float]:
    """Return raw moments E[Z^n] for the NEF-GHS base variable Z."""
    if abs(theta) >= 1.0:
        raise ValueError("nef_ghs requires |theta| < 1")
    lim = quad_limit_value(limit)
    key = (round(theta, 12), round(kappa, 12), lim)
    cached = _NEF_GHS_MOM_CACHE.get(key)
    if cached is not None:
        return cached

    psi_theta = _nef_ghs_log_partition(theta, kappa, limit)
    lower = -NEF_GHS_XMAX
    upper = NEF_GHS_XMAX

    def base_density(z: float) -> float:
        """Return NEF-GHS base density at z under canonical theta."""
        return math.exp(theta * z + _ghs_log_pdf(z, kappa) - psi_theta)

    moments = []
    for power in (1, 2, 3, 4):
        integrand = lambda z, p=power: (z**p) * base_density(z)
        val, _ = quad(
            integrand,
            lower,
            upper,
            limit=lim,
            epsabs=1e-9,
            epsrel=1e-9,
        )
        moments.append(val)

    result = tuple(moments)
    _NEF_GHS_MOM_CACHE[key] = result
    return result


def nef_ghs_params(
    kappa: float,
    theta: float,
    sigma: float,
    sqrt_t: float,
    r: float,
    q: float,
    t: float,
) -> Tuple[float, float, float]:
    """Return (mu, scale, theta) for NEF-GHS log returns."""
    if kappa <= 0.0:
        raise ValueError("nef_ghs requires kappa>0")
    if abs(theta) >= 1.0:
        raise ValueError("nef_ghs requires |theta| < 1")
    moments = nef_ghs_base_moments(theta, kappa, NEF_GHS_MGF_LIMIT)
    mean_base, second_base, _, _ = moments
    var_base = second_base - mean_base * mean_base
    if var_base <= 0.0:
        raise ValueError("nef_ghs variance evaluation failed")
    scale = sigma * sqrt_t / math.sqrt(var_base)
    if scale <= 0.0:
        raise ValueError("nef_ghs scale must be positive")
    if abs(theta + scale) >= 1.0:
        raise ValueError("nef_ghs requires |theta+scale| < 1 so mgf(1) exists")
    psi_theta = _nef_ghs_log_partition(theta, kappa, NEF_GHS_MGF_LIMIT)
    psi_theta_scale = _nef_ghs_log_partition(theta + scale, kappa, NEF_GHS_MGF_LIMIT)
    mu = (r - q) * t - (psi_theta_scale - psi_theta)
    return mu, scale, theta


def nef_ghs_log_pdf(x: float, kappa: float, theta: float, mu: float, scale: float) -> float:
    """Return log pdf for NEF-GHS."""
    z = (x - mu) / scale
    if abs(theta) >= 1.0:
        raise ValueError("nef_ghs requires |theta| < 1")
    psi_theta = _nef_ghs_log_partition(theta, kappa, NEF_GHS_MGF_LIMIT)
    return _ghs_log_pdf(z, kappa) - math.log(scale) + theta * z - psi_theta


def nef_ghs_pdf(x: float, kappa: float, theta: float, mu: float, scale: float) -> float:
    """NEF-GHS pdf with location mu and scale."""
    return math.exp(nef_ghs_log_pdf(x, kappa, theta, mu, scale))


def nef_ghs_tail_prob(
    x0: float,
    kappa: float,
    theta: float,
    mu: float,
    scale: float,
    limit: int,
) -> float:
    """Tail probability P(X>x0) for NEF-GHS."""
    lim = quad_limit_value(limit)
    val, _ = quad(
        lambda x: math.exp(nef_ghs_log_pdf(x, kappa, theta, mu, scale)),
        x0,
        math.inf,
        limit=lim,
        epsabs=1e-9,
        epsrel=1e-9,
    )
    return val


def nef_ghs_tilted_tail(
    x0: float,
    kappa: float,
    theta: float,
    mu: float,
    scale: float,
    limit: int,
) -> float:
    """Return E[exp(X) 1_{X>x0}] for NEF-GHS."""
    lim = quad_limit_value(limit)
    val, _ = quad(
        lambda x: math.exp(x + nef_ghs_log_pdf(x, kappa, theta, mu, scale)),
        x0,
        math.inf,
        limit=lim,
        epsabs=1e-9,
        epsrel=1e-9,
    )
    return val


def nef_ghs_mgf(u: float, kappa: float, theta: float, mu: float, scale: float) -> float:
    """Moment-generating function E[exp(u X)] for NEF-GHS."""
    arg = theta + scale * u
    if abs(theta) >= 1.0 or abs(arg) >= 1.0:
        raise ValueError("nef_ghs requires |theta+scale*u| < 1 for mgf")
    psi_theta = _nef_ghs_log_partition(theta, kappa, NEF_GHS_MGF_LIMIT)
    psi_arg = _nef_ghs_log_partition(arg, kappa, NEF_GHS_MGF_LIMIT)
    return math.exp(mu * u + psi_arg - psi_theta)


def _gsh_den(u: float, a_val: float) -> float:
    """Return denominator helper for GSH densities."""
    exp_u = math.exp(u)
    return exp_u * exp_u + 2.0 * a_val * exp_u + 1.0


def _gsh_log_den(u: float, a_val: float) -> float:
    """Return log denominator for stability."""
    if u >= 0.0:
        exp_neg_u = math.exp(-u)
        return 2.0 * u + math.log1p(2.0 * a_val * exp_neg_u + exp_neg_u * exp_neg_u)
    exp_u = math.exp(u)
    return math.log1p(2.0 * a_val * exp_u + exp_u * exp_u)


def _gsh_integrand(u: float, a_val: float) -> float:
    """Return stabilized base integrand for normalization."""
    if u >= 0.0:
        exp_neg_u = math.exp(-u)
        den = 1.0 + 2.0 * a_val * exp_neg_u + exp_neg_u * exp_neg_u
        return exp_neg_u / den
    exp_u = math.exp(u)
    den = exp_u * exp_u + 2.0 * a_val * exp_u + 1.0
    return exp_u / den


def gsh_constants(t: float, quad_limit: int) -> Tuple[float, float, float]:
    """Return (c1, c2, a) for the GSH distribution parameter t."""
    lim = quad_limit_value(quad_limit)
    key = (round(t, 12), lim)
    cached = _GSH_CONST_CACHE.get(key)
    if cached:
        return cached
    if t <= 0.0:
        if not (-math.pi < t <= 0.0):
            raise ValueError("gsh requires t in (-pi, 0]")
        a_val = math.cos(t)
    else:
        a_val = math.cosh(t)

    def base_integrand(u: float) -> float:
        return _gsh_integrand(u, a_val)

    norm, _ = quad(
        base_integrand,
        -math.inf,
        math.inf,
        limit=lim,
        epsabs=1e-9,
        epsrel=1e-9,
    )
    if norm <= 0.0:
        raise ValueError("gsh normalization failed")
    k_norm = 1.0 / norm
    second_moment_raw, _ = quad(
        lambda u: (u * u) * base_integrand(u),
        -math.inf,
        math.inf,
        limit=lim,
        epsabs=1e-9,
        epsrel=1e-9,
    )
    var_base = k_norm * second_moment_raw
    if var_base <= 0.0:
        raise ValueError("gsh variance calibration failed")
    c2 = math.sqrt(var_base)
    c1 = c2 * k_norm
    _GSH_CONST_CACHE[key] = (c1, c2, a_val)
    return c1, c2, a_val


def gsh_log_pdf(x: float, t: float, quad_limit: int) -> float:
    """Return log pdf of the GSH distribution."""
    c1, c2, a_val = gsh_constants(t, quad_limit)
    u = c2 * x
    return math.log(c1) + u - _gsh_log_den(u, a_val)


def gsh_pdf(x: float, t: float, quad_limit: int) -> float:
    """Return the GSH pdf."""
    return math.exp(gsh_log_pdf(x, t, quad_limit))


def sgsh_log_pdf(x: float, t: float, skew: float, quad_limit: int) -> float:
    """Return log pdf for the SGSH distribution."""
    if skew <= 0.0:
        raise ValueError("sgsh skew parameter must be > 0")
    norm = 2.0 / (skew + 1.0 / skew)
    if x < 0.0:
        return math.log(norm) - math.log(skew) + gsh_log_pdf(x / skew, t, quad_limit)
    return math.log(norm) + math.log(skew) + gsh_log_pdf(skew * x, t, quad_limit)


def sgsh_pdf(x: float, t: float, skew: float, quad_limit: int) -> float:
    """Return SGSH pdf."""
    return math.exp(sgsh_log_pdf(x, t, skew, quad_limit))


def gsh_mgf(u: float, t: float, quad_limit: int) -> float:
    """Return mgf of the standardized GSH distribution."""
    if abs(u) < 1e-12:
        return 1.0
    lim = quad_limit_value(quad_limit)
    key = (round(u, 12), round(t, 12), lim)
    cached = _GSH_MGF_CACHE.get(key)
    if cached is not None:
        return cached

    def integrand(x: float) -> float:
        return math.exp(u * x + gsh_log_pdf(x, t, quad_limit))

    val, _ = quad(
        integrand,
        -math.inf,
        math.inf,
        limit=lim,
        epsabs=1e-9,
        epsrel=1e-9,
    )
    _GSH_MGF_CACHE[key] = val
    return val


def gsh_abs_moment(t: float, quad_limit: int) -> float:
    """Return E[|X|] for the standardized GSH distribution."""
    lim = quad_limit_value(quad_limit)
    key = (round(t, 12), lim)
    cached = _GSH_ABS_CACHE.get(key)
    if cached is not None:
        return cached

    def integrand(x: float) -> float:
        return x * gsh_pdf(x, t, quad_limit)

    val, _ = quad(
        integrand,
        0.0,
        math.inf,
        limit=lim,
        epsabs=1e-9,
        epsrel=1e-9,
    )
    abs_moment = 2.0 * val
    _GSH_ABS_CACHE[key] = abs_moment
    return abs_moment


def sgsh_mgf(u: float, t: float, skew: float, quad_limit: int) -> float:
    """Return mgf of the standardized SGSH distribution."""
    if abs(u) < 1e-12:
        return 1.0
    lim = quad_limit_value(quad_limit)
    key = (round(u, 12), round(t, 12), round(skew, 12), lim)
    cached = _SGSH_MGF_CACHE.get(key)
    if cached is not None:
        return cached

    def integrand(x: float) -> float:
        return math.exp(u * x + sgsh_log_pdf(x, t, skew, quad_limit))

    val, _ = quad(
        integrand,
        -math.inf,
        math.inf,
        limit=lim,
        epsabs=1e-9,
        epsrel=1e-9,
    )
    _SGSH_MGF_CACHE[key] = val
    return val


def sgsh_base_moments(t: float, skew: float, quad_limit: int) -> Tuple[float, float]:
    """Return (mean, variance) of the base SGSH distribution."""
    if skew <= 0.0:
        raise ValueError("sgsh skew must be > 0")
    lim = quad_limit_value(quad_limit)
    key = (round(t, 12), round(skew, 12), lim)
    cached = _SGSH_MOMENT_CACHE.get(key)
    if cached:
        return cached
    abs_m = gsh_abs_moment(t, quad_limit)
    lam = skew
    lam_inv = 1.0 / lam
    denom = lam + lam_inv
    mean = (lam_inv - lam) / denom * abs_m
    second = (lam * lam + lam_inv * lam_inv) / denom
    var = max(second - mean * mean, 0.0)
    _SGSH_MOMENT_CACHE[key] = (mean, var)
    return mean, var


def sgsh_base_high_moments(
    t: float,
    skew: float,
    quad_limit: int,
) -> Tuple[float, float]:
    """Return (third_central, fourth_central) moments for SGSH base."""
    if skew <= 0.0:
        raise ValueError("sgsh skew must be > 0")
    lim = quad_limit_value(quad_limit)
    key = (round(t, 12), round(skew, 12), lim)
    cached = _SGSH_HIGHER_CACHE.get(key)
    if cached:
        return cached
    mean_base, _ = sgsh_base_moments(t, skew, quad_limit)

    def pdf_val(x: float) -> float:
        return math.exp(sgsh_log_pdf(x, t, skew, quad_limit))

    third_neg, _ = quad(
        lambda x: ((x - mean_base) ** 3) * pdf_val(x),
        -math.inf,
        0.0,
        limit=lim,
        epsabs=1e-8,
        epsrel=1e-8,
    )
    third_pos, _ = quad(
        lambda x: ((x - mean_base) ** 3) * pdf_val(x),
        0.0,
        math.inf,
        limit=lim,
        epsabs=1e-8,
        epsrel=1e-8,
    )
    fourth_neg, _ = quad(
        lambda x: ((x - mean_base) ** 4) * pdf_val(x),
        -math.inf,
        0.0,
        limit=lim,
        epsabs=1e-8,
        epsrel=1e-8,
    )
    fourth_pos, _ = quad(
        lambda x: ((x - mean_base) ** 4) * pdf_val(x),
        0.0,
        math.inf,
        limit=lim,
        epsabs=1e-8,
        epsrel=1e-8,
    )
    third = third_neg + third_pos
    fourth = fourth_neg + fourth_pos
    _SGSH_HIGHER_CACHE[key] = (third, fourth)
    return third, fourth


def gsh_params(
    param_t: float,
    sigma: float,
    sqrt_t: float,
    r: float,
    q: float,
    t_mat: float,
    quad_limit: int,
) -> Tuple[float, float]:
    """Return (mu, scale) for the GSH log returns."""
    scale = sigma * sqrt_t
    mgf_val = gsh_mgf(scale, param_t, quad_limit)
    if mgf_val <= 0.0 or not math.isfinite(mgf_val):
        raise ValueError("gsh mgf evaluation failed")
    mu = (r - q) * t_mat - math.log(mgf_val)
    return mu, scale


def sgsh_params(
    param_t: float,
    skew: float,
    sigma: float,
    sqrt_t: float,
    r: float,
    q: float,
    t_mat: float,
    quad_limit: int,
) -> Tuple[float, float]:
    """Return (mu, scale) for SGSH log returns."""
    if skew <= 0.0:
        raise ValueError("sgsh skew must be > 0")
    _, var_base = sgsh_base_moments(param_t, skew, quad_limit)
    if var_base <= 0.0:
        raise ValueError("sgsh base variance must be positive")
    scale = sigma * sqrt_t / math.sqrt(var_base)
    mgf_val = sgsh_mgf(scale, param_t, skew, quad_limit)
    if mgf_val <= 0.0 or not math.isfinite(mgf_val):
        raise ValueError("sgsh mgf evaluation failed")
    mu = (r - q) * t_mat - math.log(mgf_val)
    return mu, scale


def gh_mgf(scale: float, lam: float, alpha: float, beta: float, limit: int) -> float:
    """Return E[exp(scale*Z)] for Z~GH(lambda,alpha,beta,loc=0,scale=1)."""
    if abs(beta + scale) >= alpha:
        raise ValueError("generalized hyperbolic requires |beta+scale| < alpha for mgf evaluation")
    lim = quad_limit_value(limit)
    integrand = lambda z: math.exp(scale * z) * genhyperbolic.pdf(z, lam, alpha, beta)
    val, _ = quad(
        integrand,
        -GH_ZMAX,
        GH_ZMAX,
        limit=lim,
        epsabs=1e-9,
        epsrel=1e-9,
    )
    return val


@lru_cache(maxsize=None)
def gh_base_stats(lam: float, alpha: float, beta: float) -> Tuple[float, float, float, float]:
    """Return base mean/var/skew/kurt for GH(lambda,alpha,beta,loc=0,scale=1)."""
    mean, var, skew, kurt_excess = genhyperbolic.stats(lam, alpha, beta, moments="mvsk")
    if (
        not math.isfinite(mean)
        or not math.isfinite(var)
        or not math.isfinite(skew)
        or not math.isfinite(kurt_excess)
        or var <= 0.0
    ):
        raise ValueError("generalized hyperbolic stats are not finite; adjust parameters")
    return float(mean), float(var), float(skew), float(kurt_excess + 3.0)


def gh_params(
    lam: float,
    alpha: float,
    beta: float,
    sigma: float,
    sqrt_t: float,
    r: float,
    q: float,
    t: float,
    quad_limit: int,
) -> Tuple[float, float]:
    """Return (loc, scale) for the generalized hyperbolic log returns."""
    if alpha <= 0.0:
        raise ValueError("generalized hyperbolic alpha must be > 0")
    if abs(beta) >= alpha:
        raise ValueError("generalized hyperbolic requires |beta| < alpha")
    if t <= 0.0 or sigma <= 0.0:
        raise ValueError("generalized hyperbolic requires sigma>0 and t>0")
    _, var_b, _, _ = gh_base_stats(lam, alpha, beta)
    std_base = math.sqrt(var_b)
    scale = sigma * sqrt_t / std_base
    if scale <= 0.0:
        raise ValueError("generalized hyperbolic scale must be positive")
    if abs(beta + scale) >= alpha:
        raise ValueError("generalized hyperbolic requires |beta+scale| < alpha for mgf at 1")
    mgf1 = gh_mgf(scale, lam, alpha, beta, quad_limit)
    if mgf1 <= 0.0 or not math.isfinite(mgf1):
        raise ValueError("generalized hyperbolic mgf evaluation failed")
    loc = (r - q) * t - math.log(mgf1)
    return loc, scale


def gh_pdf(x: float, lam: float, alpha: float, beta: float, loc: float, scale: float) -> float:
    """GH pdf evaluated at x."""
    if scale <= 0.0:
        raise ValueError("generalized hyperbolic scale must be positive")
    z = (x - loc) / scale
    return genhyperbolic.pdf(z, lam, alpha, beta) / scale


def gh_tail_prob(
    x0: float,
    lam: float,
    alpha: float,
    beta: float,
    loc: float,
    scale: float,
) -> float:
    """Tail probability P(X>x0) for GH."""
    if scale <= 0.0:
        raise ValueError("generalized hyperbolic scale must be positive")
    z = (x0 - loc) / scale
    return genhyperbolic.sf(z, lam, alpha, beta)


def gh_tilted_tail(
    x0: float,
    lam: float,
    alpha: float,
    beta: float,
    loc: float,
    scale: float,
    limit: int,
) -> float:
    """Return E[exp(X) 1_{X>x0}] for GH."""
    if scale <= 0.0:
        raise ValueError("generalized hyperbolic scale must be positive")
    lim = quad_limit_value(limit)
    z0 = (x0 - loc) / scale
    exp_loc = math.exp(loc)

    def integrand(z: float) -> float:
        """Integrand for tilted GH expectation over base variable z."""
        return math.exp(scale * z) * genhyperbolic.pdf(z, lam, alpha, beta)

    if z0 >= GH_ZMAX:
        return 0.0
    lower = max(z0, -GH_ZMAX)
    val, _ = quad(
        integrand,
        lower,
        GH_ZMAX,
        limit=lim,
        epsabs=1e-9,
        epsrel=1e-9,
    )
    return exp_loc * val


def _cgmy_variance_unit(
    C_val: float,
    G_val: float,
    M_val: float,
    Y_val: float,
    gamma_neg: float,
) -> float:
    """Return the per-unit-time variance of an unscaled CGMY process."""
    coeff = C_val * gamma_neg * Y_val * (Y_val - 1.0)
    return coeff * (M_val ** (Y_val - 2.0) + G_val ** (Y_val - 2.0))


def cgmy_params(
    C_val: float,
    G_val: float,
    M_val: float,
    Y_val: float,
    sigma: float,
    sqrt_t: float,
    r: float,
    q: float,
    t: float,
) -> CgmyParams:
    """Return calibrated CGMY parameters that match the target variance."""
    del sqrt_t  # unused but retained for signature consistency
    if C_val <= 0.0 or G_val <= 0.0 or M_val <= 0.0:
        raise ValueError("cgmy requires positive C, G, and M")
    if not (0.0 < Y_val < 2.0):
        raise ValueError("cgmy Y must lie in (0,2) for finite variance")
    if sigma <= 0.0:
        raise ValueError("cgmy requires sigma>0")
    gamma_neg = math.gamma(-Y_val)
    if not math.isfinite(gamma_neg):
        raise ValueError("cgmy gamma(-Y) is not finite; adjust Y")
    var_unit = _cgmy_variance_unit(C_val, G_val, M_val, Y_val, gamma_neg)
    if var_unit <= 0.0:
        raise ValueError("cgmy variance calibration failed; check parameters")
    scale = sigma / math.sqrt(var_unit)
    if scale <= 0.0:
        raise ValueError("cgmy scale must be positive")
    mgf_bound = min(G_val, M_val)
    if scale >= mgf_bound:
        raise ValueError("cgmy requires sigma small enough that scale < min(G,M)")
    mu_shift = (r - q) * t - C_val * gamma_neg * t * (
        (M_val - scale) ** Y_val
        - (M_val ** Y_val)
        + (G_val + scale) ** Y_val
        - (G_val ** Y_val)
    )
    return CgmyParams(C_val, G_val, M_val, Y_val, scale, mu_shift, gamma_neg, t)


def _cgmy_complex_power(base: complex, power: float) -> complex:
    """Return base**power handling complex arguments."""
    return cmath.exp(power * cmath.log(base))


def cgmy_characteristic(z: complex, params: CgmyParams) -> complex:
    """Characteristic function of the CGMY log-return distribution."""
    c_val, g_val, m_val, y_val, scale, mu_shift, gamma_neg, t = params
    arg_pos = g_val + 1j * scale * z
    arg_neg = m_val - 1j * scale * z
    term = (
        _cgmy_complex_power(arg_neg, y_val)
        - (m_val ** y_val)
        + _cgmy_complex_power(arg_pos, y_val)
        - (g_val ** y_val)
    )
    exponent = 1j * z * mu_shift + c_val * gamma_neg * t * term
    return cmath.exp(exponent)


def cgmy_mgf(u: float, params: CgmyParams) -> float:
    """Moment-generating function E[exp(uX)] for CGMY log returns."""
    c_val, g_val, m_val, y_val, scale, mu_shift, gamma_neg, t = params
    if scale * u >= min(g_val, m_val):
        raise ValueError("cgmy requires scale*u < min(G,M) for finite mgf")
    term = (
        (m_val - scale * u) ** y_val
        - (m_val ** y_val)
        + (g_val + scale * u) ** y_val
        - (g_val ** y_val)
    )
    exponent = mu_shift * u + c_val * gamma_neg * t * term
    return math.exp(exponent)


def cgmy_pdf(x: float, params: CgmyParams, quad_limit: int) -> float:
    """Return the CGMY pdf at x via Fourier inversion."""
    lim = quad_limit_value(quad_limit)

    def integrand(u: float) -> float:
        """Return integrand for the cosine inversion."""
        phi_val = cgmy_characteristic(u, params)
        val = cmath.exp(-1j * u * x) * phi_val
        return val.real

    val, _ = quad(
        integrand,
        0.0,
        CGMY_U_MAX,
        limit=lim,
        epsabs=1e-6,
        epsrel=1e-6,
    )
    density = max(val / math.pi, 0.0)
    return density


def nig_delta(variance: float, alpha: float, beta: float) -> float:
    """Compute delta to match the target variance."""
    num = variance * (alpha * alpha - beta * beta) ** 1.5
    den = alpha * alpha
    return num / den


def nig_pdf(x: float, alpha: float, beta: float, delta: float, mu: float) -> float:
    """NIG pdf."""
    arg = math.sqrt(delta * delta + (x - mu) * (x - mu))
    arg = max(arg, 1e-12)
    k_val = kv(1, alpha * arg)
    coef = alpha * delta / math.pi
    expo = math.exp(delta * math.sqrt(alpha * alpha - beta * beta) + beta * (x - mu))
    return coef * expo * k_val / arg


def nig_tail_prob(
    x0: float,
    alpha: float,
    beta: float,
    delta: float,
    mu: float,
    limit: int,
) -> float:
    """Tail probability P(X>x0) for NIG."""
    lim = quad_limit_value(limit)
    if x0 >= NIG_XMAX:
        return 0.0
    if x0 <= -NIG_XMAX:
        return 1.0
    lower = max(x0, -NIG_XMAX)
    val, _ = quad(
        lambda x: nig_pdf(x, alpha, beta, delta, mu),
        lower,
        NIG_XMAX,
        limit=lim,
        epsabs=1e-12,
        epsrel=1e-12,
    )
    return val


def nig_tilted_tail(
    x0: float,
    alpha: float,
    beta: float,
    delta: float,
    mu: float,
    limit: int,
) -> float:
    """Return E[exp(X) 1_{X>x0}] for NIG."""
    lim = quad_limit_value(limit)
    if x0 >= NIG_XMAX:
        return 0.0
    lower = max(x0, -NIG_XMAX)
    val, _ = quad(
        lambda x: math.exp(x) * nig_pdf(x, alpha, beta, delta, mu),
        lower,
        NIG_XMAX,
        limit=lim,
        epsabs=1e-12,
        epsrel=1e-12,
    )
    return val


def vg_call_expectation(
    s0: float,
    k: float,
    mu: float,
    theta: float,
    sigma_vg: float,
    shape: float,
    scale: float,
    limit: int,
) -> float:
    """Expectation of (S0*exp(X)-K)+ under variance gamma via gamma mixing."""
    if shape <= 0.0 or scale <= 0.0:
        raise ValueError("variance gamma requires shape>0 and scale>0")
    log_norm = math.lgamma(shape) + shape * math.log(scale)
    log_k_ratio = math.log(k / s0)
    upper = max(50.0, shape * scale * VG_G_SCALE_FACTOR)
    lim = quad_limit_value(limit)

    def gamma_pdf(g: float) -> float:
        g_pos = max(g, VG_G_MIN)
        return math.exp((shape - 1.0) * math.log(g_pos) - g_pos / scale - log_norm)

    def integrand(g: float) -> float:
        if g < 0.0:
            g = 0.0
        pdf = gamma_pdf(g)
        m = mu + theta * g
        v = sigma_vg * math.sqrt(g)
        if v == 0.0:
            payoff = max(s0 * math.exp(m) - k, 0.0)
        else:
            v2 = v * v
            d1 = (m + v2 - log_k_ratio) / v
            d2 = d1 - v
            payoff = s0 * math.exp(m + 0.5 * v2) * norm.cdf(d1) - k * norm.cdf(d2)
        return payoff * pdf

    val1, _ = quad(integrand, 0.0, upper, limit=lim, epsabs=1e-7, epsrel=1e-7)
    val2, _ = quad(integrand, upper, math.inf, limit=lim, epsabs=1e-7, epsrel=1e-7)
    return val1 + val2


def vg_pdf(
    x: float,
    mu: float,
    theta: float,
    sigma_vg: float,
    shape: float,
    scale: float,
) -> float:
    """Approximate VG pdf via truncated gamma mixture."""
    if shape <= 0.0 or scale <= 0.0:
        raise ValueError("variance gamma requires shape>0 and scale>0")
    log_norm = math.lgamma(shape) + shape * math.log(scale)
    total = 0.0

    for n in range(VG_PDF_GTERMS):
        g = scale * (n + 0.5)
        log_pdf = (shape - 1.0) * math.log(g) - g / scale - log_norm
        pdf_g = math.exp(log_pdf)
        mean = mu + theta * g
        var = sigma_vg * sigma_vg * g
        if var <= 0.0:
            density = 0.0
        else:
            density = (1.0 / math.sqrt(2.0 * math.pi * var)) * math.exp(-0.5 * (x - mean) ** 2 / var)
        total += pdf_g * density
    return total
