"""Compare European option prices and implied vols across diverse log-return distributions."""

import math
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from distributions import tukey_lambda_params
from optionpricingrecipe import (
    implied_vol_from_call,
    log_return_stats,
    option_price,
    plot_implied_vols,
    plot_terminal_distributions,
    terminal_price_stats,
)

PLOT_DISTRIBUTIONS = False
PLOT_IMPLIED_VOLS = True
PLOT_DISTRIBUTIONS_FILE: Optional[str] = "density.png"
PLOT_IMPLIED_VOLS_FILE: Optional[str] = "vol.png"
PRICES_CSV_FILE: Optional[str] = "option_prices.csv"
IMPLIED_VOL_CSV_FILE: Optional[str] = "implied_vols.csv"

def main() -> None:
    """Drive the option pricer demo using the shared pricing modules."""
    s0 = 100.0 # initial stock price
    r = 0.03 # risk-free rate
    q = 0.00 # dividend yield
    t = 1.0 # time to expiration in years
    sigma = 0.20 # annualized volatility
    quad_limit = 1000 # cap adaptive quad integrations to keep slow densities tractable
    sqrt_t = math.sqrt(t)
    disc_r = math.exp(-r * t)
    disc_q = math.exp(-q * t)

    max_dist = 3 # None # max number of probability distributions to study -- set to None for no max
    strike_min = 70.0
    strike_count = 61
    strike_step = 1.0
    strikes = np.arange(strike_min, strike_min + strike_count * strike_step, strike_step)
    log_k_min = float(math.log(float(np.min(strikes)) / s0))
    log_k_max = float(math.log(float(np.max(strikes)) / s0))
    # Table stride controls which strikes print: 1=every strike, 3=every third, <1 hides table.
    strike_table_stride = 5
    # Option type for comparison: "c" (call), "p" (put), or "straddle".
    option_type = "straddle"
    # Champernowne shape d trades kurtosis: negative d heavier tails, positive d thinner.
    champer_d_list = [-1.0, 0.0, 1.0, 2.0, 3.0]
    # GED power controls peak/tail weight: p=1 Laplace heavy tails, p=2 Gaussian.
    ged_powers = [1.0, 1.5, 2.0]
    # NIG beta shifts skewness (negative beta -> left tail); alpha fixes overall tail steepness.
    nig_betas = np.array([-1.5, -0.75, 0.0], dtype=float)
    nig_alpha = 5.0
    # VG theta introduces skew; nu (variance of subordinator) widens tails.
    vg_thetas = np.array([-0.1, -0.05, 0.0], dtype=float)
    vg_nu = 0.2
    # Hyperbolic (alpha,beta) pairs: alpha tightens tails, beta sets asymmetry.
    hyper_shapes = [(10.0, 0.0), (12.0, -1.0), (9.0, 1.0)]
    # Johnson SU (a,b): a shifts skew, b stretches kurtosis.
    johnson_shapes = [(0.0, 4.0), (0.5, 5.0), (-0.5, 6.0)]
    # Tukey lambda: smaller lambda -> heavier tails; larger lambda -> thinner.
    tukey_lambdas = [0.1, 0.25, 0.5]
    # GH (lambda,alpha,beta): lambda tunes tail type, beta skew, alpha tail decay.
    gh_shapes = [(1.0, 2.0, 0.0), (0.75, 2.5, -0.5), (1.25, 3.0, 0.5)]
    # NEF-GHS (kappa controls kurtosis, lam -> theta controls skew).
    nef_ghs_shapes = [(1.0, 0.0), (1.5, 0.5), (0.8, -0.5)]
    # GSH t parameter slides kurtosis from heavy (negative) to light (positive).
    gsh_t_values = [-2.5, -1.0, 0.0, 0.5]
    # SGSH pairs (t, skew): t for tail weight, skew>1 right skew / <1 left skew.
    sgsh_shapes = [(0.0, 1.1), (0.0, 1.0)]
    # CGMY tuples: C scales jump intensity, G vs M set negative/positive damping (skew sign comes from which is larger), Y tunes tail thickness (<2 heavy).
    cgmy_shapes = [(1.0, 5.0, 5.0, 0.1), (0.9, 1.0, 2.0, 0.1), (0.7, 1.8, 11.4, 1.15)]
    requested = [
         "normal",
         "hyperbolic",
         "nig",
         "hypsecant",
         "laplace",
         "logistic",
         "johnsonsu",
         "tukeylambda",
         "champernowne",
         "ged",
         "vg",
         "gh",
         "nefghs",
         "gsh",
         "sgsh",
     ]  # shorten this list (e.g. ["normal","vg"]) to focus on a subset
    if max_dist is not None:
        requested = requested[:max_dist]
    print("requested:", requested) # debug
    option_names = {"c": "call", "p": "put", "straddle": "straddle"}
    option_label = option_names.get(option_type, option_type)
    print(f"European {option_label} comparison")
    print(f"s0={s0:.4f} r={r:.4f} q={q:.4f} t={t:.4f} sigma={sigma:.4f}")
    print(f"Champernowne d values: {', '.join(f'{d:g}' for d in champer_d_list)}")
    print(f"GED powers: {', '.join(f'{p:g}' for p in ged_powers)}")
    print(f"NIG beta skew params: {', '.join(f'{b:g}' for b in nig_betas)}")
    print(f"VG theta skew params: {', '.join(f'{th:g}' for th in vg_thetas)} (nu={vg_nu:g})")
    print(
        "Hyperbolic (alpha,beta): "
        + ", ".join(f"({alpha:g},{beta:g})" for alpha, beta in hyper_shapes)
    )
    print(
        "Johnson SU (a,b): "
        + ", ".join(f"({a:g},{b:g})" for a, b in johnson_shapes)
    )
    tukey_info: list[tuple[float, str]] = []
    valid_tukey: list[float] = []
    tukey_skipped: list[str] = []
    for lam in tukey_lambdas:
        try:
            loc, scale = tukey_lambda_params(lam, sigma, sqrt_t, r, q, t, quad_limit)
            if lam > 0.0:
                log_min = loc - scale / lam
                log_max = loc + scale / lam
                strike_min_support = s0 * math.exp(log_min)
                strike_max_support = s0 * math.exp(log_max)
                tukey_info.append(
                    (lam, f"{lam:g} (S_T in [{strike_min_support:.1f}, {strike_max_support:.1f}])")
                )
                if log_min <= log_k_min + 1e-8 and log_max >= log_k_max - 1e-8:
                    valid_tukey.append(lam)
                else:
                    tukey_skipped.append(
                        f"{lam:g} (log range [{log_min:.3f},{log_max:.3f}] vs strikes [{log_k_min:.3f},{log_k_max:.3f}])"
                    )
            else:
                tukey_info.append((lam, f"{lam:g} (heavy tails)"))
                valid_tukey.append(lam)
        except Exception as exc:
            tukey_info.append((lam, f"{lam:g} (calibration failed: {exc})"))
            tukey_skipped.append(f"{lam:g} ({exc})")

    tukey_desc = ", ".join(info for _, info in tukey_info) or "none"
    print("Tukey lambda shapes: " + tukey_desc)
    if tukey_skipped:
        print(
            "  Tukey shapes dropped because log-return support misses strike grid: "
            + ", ".join(tukey_skipped)
        )
    print(
        "Gen. hyperbolic (lam,alpha,beta): "
        + ", ".join(f"({lam:g},{alpha:g},{beta:g})" for lam, alpha, beta in gh_shapes)
    )
    print(
        "NEF-GHS (kappa,lam): "
        + ", ".join(f"({kappa:g},{lam:g})" for kappa, lam in nef_ghs_shapes)
    )
    print(f"GSH t values: {', '.join(f'{val:g}' for val in gsh_t_values)}")
    print(
        "SGSH (t,skew): "
        + ", ".join(f"({t_val:g},{skew:g})" for t_val, skew in sgsh_shapes)
    )
    print(
        "CGMY (C,G,M,Y): "
        + ", ".join(f"({C_val:g},{G_val:g},{M_val:g},{Y_val:g})" for C_val, G_val, M_val, Y_val in cgmy_shapes)
    )
    print("")

    dist_columns: List[Tuple[str, str, Optional[dict[str, float]]]] = []
    if "normal" in requested:
        dist_columns.append(("normal", "normal", None))
    if "hypsecant" in requested:
        dist_columns.append(("hypsecant", "hypsecant", None))
    if "logistic" in requested:
        dist_columns.append(("logistic", "logistic", None))
    if "laplace" in requested:
        dist_columns.append(("laplace", "laplace", None))
    if "hyperbolic" in requested:
        dist_columns.extend(
            (
                f"hyp_a={alpha:g}_b={beta:g}",
                "hyperbolic",
                {"alpha": alpha, "beta": beta},
            )
            for alpha, beta in hyper_shapes
        )
    if "johnsonsu" in requested:
        dist_columns.extend(
            (
                f"jsu_a={a:g}_b={b:g}",
                "johnsonsu",
                {"a": a, "b": b},
            )
            for a, b in johnson_shapes
        )
    if "tukeylambda" in requested:
        dist_columns.extend(
            (f"tl_{lam:g}", "tukeylambda", {"lam": lam}) for lam in valid_tukey
        )
    if "champernowne" in requested:
        dist_columns.extend(
            (f"ch_d={d:g}", "champernowne", {"d": d}) for d in champer_d_list
        )
    if "ged" in requested:
        dist_columns.extend(
            (f"ged_p={p:g}", "ged", {"p": p}) for p in ged_powers
        )
    if "nig" in requested:
        dist_columns.extend(
            (f"nig_b={beta:g}", "nig", {"alpha": nig_alpha, "beta": float(beta)})
            for beta in nig_betas
        )
    if "vg" in requested:
        dist_columns.extend(
            (f"vg_theta={theta:g}", "vg", {"theta": float(theta), "nu": vg_nu})
            for theta in vg_thetas
        )
    if "nefghs" in requested:
        dist_columns.extend(
            (
                f"nef_k={kappa:g}_lam={lam:g}",
                "nefghs",
                {"kappa": kappa, "lam": lam},
            )
            for kappa, lam in nef_ghs_shapes
        )
    if "gsh" in requested:
        dist_columns.extend(
            (f"gsh_t={t_val:g}", "gsh", {"t": t_val}) for t_val in gsh_t_values
        )
    if "sgsh" in requested:
        dist_columns.extend(
            (
                f"sgsh_t={t_val:g}_s={skew:g}",
                "sgsh",
                {"t": t_val, "skew": skew},
            )
            for t_val, skew in sgsh_shapes
        )
    if "cgmy" in requested:
        dist_columns.extend(
            (
                f"cgmy_C={C_val:g}_G={G_val:g}_M={M_val:g}_Y={Y_val:g}",
                "cgmy",
                {"C": C_val, "G": G_val, "M": M_val, "Y": Y_val},
            )
            for C_val, G_val, M_val, Y_val in cgmy_shapes
        )
    if "gh" in requested:
        dist_columns.extend(
            (
                f"gh_l={lam:g}_a={alpha:g}_b={beta:g}",
                "gh",
                {"lam": lam, "alpha": alpha, "beta": beta},
            )
            for lam, alpha, beta in gh_shapes
        )

    min_width = 14
    max_label_len = max((len(label) for label, *_ in dist_columns), default=min_width)
    col_width = max(min_width, max_label_len + 2)
    label_width = max(15, max_label_len + 2)
    header = "   k " + "".join(f"{label:>{col_width}s}" for label, *_ in dist_columns)
    show_tables = strike_table_stride >= 1
    if show_tables:
        print(header)
        print("-" * len(header))

    rows_prices: list[str] = []
    rows_vols: list[str] = []
    price_values: list[list[float]] = []
    vol_values: list[list[float]] = []

    failures: dict[str, str] = {}
    for idx, kk in enumerate(strikes):
        kk = float(kk)
        row_price = [f"{int(kk):5d}"]
        row_vol = [f"{int(kk):5d}"]
        price_row_values: list[float] = []
        vol_row_values: list[float] = []
        parity_tol = 1e-8 * max(s0, 1.0)
        for label, dist_name, dist_params in dist_columns:
            try:
                price = option_price(
                    option_type,
                    s0,
                    kk,
                    t,
                    r,
                    sigma,
                    q=q,
                    dist=dist_name,
                    dist_params=dist_params,
                    quad_limit=quad_limit,
                )
                call_value = price
                if option_type == "p":
                    call_value = price + s0 * disc_q - kk * disc_r
                elif option_type == "straddle":
                    call_value = 0.5 * (price + s0 * disc_q - kk * disc_r)

                iv = 0.0
                adj_call = call_value
                if adj_call <= 0.0:
                    if adj_call < -parity_tol:
                        raise RuntimeError(
                            f"parity produced negative call value ({adj_call:.6g}) for {label}"
                        )
                    adj_call = parity_tol
                iv = implied_vol_from_call(adj_call, s0, kk, r, q, t)

                row_price.append(f"{price:{col_width}.6f}")
                price_row_values.append(price)
                row_vol.append(f"{iv:{col_width}.6f}")
                vol_row_values.append(iv)
            except Exception as exc:
                if label not in failures:
                    failures[label] = str(exc)
                row_price.append(f"{float('nan'):{col_width}.6f}")
                price_row_values.append(float("nan"))
                row_vol.append(f"{float('nan'):{col_width}.6f}")
                vol_row_values.append(float("nan"))

        price_str = " ".join(row_price)
        vol_str = " ".join(row_vol)
        rows_prices.append(price_str)
        rows_vols.append(vol_str)
        price_values.append(price_row_values)
        vol_values.append(vol_row_values)
        if show_tables and (idx % strike_table_stride == 0):
            print(price_str)

    if show_tables:
        print("")
        print("Implied volatility table")
        print(header)
        print("-" * len(header))
        for idx, row in enumerate(rows_vols):
            if idx % strike_table_stride == 0:
                print(row)

    labels = [label for label, *_ in dist_columns]
    vol_stats_df = None
    if PRICES_CSV_FILE and dist_columns:
        prices_df = pd.DataFrame(price_values, columns=labels)
        prices_df.insert(0, "strike", [float(val) for val in strikes])
        prices_df.to_csv(PRICES_CSV_FILE, index=False)
    if IMPLIED_VOL_CSV_FILE and dist_columns:
        vols_df = pd.DataFrame(vol_values, columns=labels)
        vols_df.insert(0, "strike", [float(val) for val in strikes])
        vols_df.to_csv(IMPLIED_VOL_CSV_FILE, index=False)
    if dist_columns:
        strikes_float = [float(val) for val in strikes]
        vol_df = pd.DataFrame(vol_values, columns=labels)
        stats_rows = []
        for label in labels:
            series = pd.to_numeric(vol_df[label], errors="coerce")
            mean_val = float(series.mean(skipna=True))
            std_val = float(series.std(skipna=True))
            min_val = float(series.min(skipna=True)) if series.notna().any() else float("nan")
            max_val = float(series.max(skipna=True)) if series.notna().any() else float("nan")
            # Compute discrete second-derivative energy to quantify smoothness.
            second_diff = []
            for idx in range(1, len(strikes_float) - 1):
                left = series.iat[idx - 1]
                mid = series.iat[idx]
                right = series.iat[idx + 1]
                # Skip if any term is nan.
                if not (np.isfinite(left) and np.isfinite(mid) and np.isfinite(right)):
                    continue
                k_left = strikes_float[idx - 1]
                k_mid = strikes_float[idx]
                k_right = strikes_float[idx + 1]
                spacing_left = k_mid - k_left
                spacing_right = k_right - k_mid
                if spacing_left <= 0.0 or spacing_right <= 0.0:
                    continue
                second_deriv = (
                    (right - mid) / spacing_right - (mid - left) / spacing_left
                ) / ((spacing_left + spacing_right) * 0.5)
                second_diff.append(second_deriv ** 2)
            smooth_energy = float(np.sum(second_diff)) if second_diff else float("nan")
            stats_rows.append(
                {
                    "dist": label,
                    "mean": mean_val,
                    "std": std_val,
                    "min": min_val,
                    "max": max_val,
                    "curvature_energy": smooth_energy,
                }
            )
        vol_stats_df = pd.DataFrame(stats_rows)
        vol_stats_df.sort_values(by="curvature_energy", inplace=True)

    print("")
    print("Log-return moments")
    stats_header = f"{'dist':<{label_width}}{'mean':>12}{'std':>12}{'skew':>12}{'kurt':>12}"
    print(stats_header)
    print("-" * len(stats_header))
    for label, dist_name, dist_params in dist_columns:
        if label in failures:
            mean = std = skew = kurt = float("nan")
        else:
            try:
                mean, std, skew, kurt = log_return_stats(
                    dist_name,
                    dist_params,
                    sigma,
                    r,
                    q,
                    t,
                    quad_limit,
                )
            except Exception as exc:
                failures.setdefault(label, str(exc))
                mean = std = skew = kurt = float("nan")
        print(f"{label:<{label_width}}{mean:12.6f}{std:12.6f}{skew:12.6f}{kurt:12.6f}")

    print("")
    print("Terminal price moments")
    st_header = f"{'dist':<{label_width}}{'mean':>12}{'std':>12}{'skew':>12}{'kurt':>12}"
    print(st_header)
    print("-" * len(st_header))
    for label, dist_name, dist_params in dist_columns:
        if label in failures:
            mean_st = std_st = skew_st = kurt_st = float("nan")
        else:
            try:
                mean_st, std_st, skew_st, kurt_st = terminal_price_stats(
                    dist_name,
                    dist_params,
                    s0,
                    sigma,
                    r,
                    q,
                    t,
                    quad_limit,
                )
            except Exception as exc:
                failures.setdefault(label, str(exc))
                mean_st = std_st = skew_st = kurt_st = float("nan")
        print(f"{label:<{label_width}}{mean_st:12.6f}{std_st:12.6f}{skew_st:12.6f}{kurt_st:12.6f}")

    valid_columns = [col for col in dist_columns if col[0] not in failures]
    if (PLOT_DISTRIBUTIONS or PLOT_DISTRIBUTIONS_FILE is not None) and valid_columns:
        plot_terminal_distributions(
            s0,
            sigma,
            r,
            q,
            t,
            valid_columns,
            quad_limit,
            outfile=PLOT_DISTRIBUTIONS_FILE,
        )
    if (PLOT_IMPLIED_VOLS or PLOT_IMPLIED_VOLS_FILE is not None) and valid_columns:
        indices = [idx for idx, label in enumerate(labels) if label not in failures]
        filtered_vols = [
            [row[i] for i in indices]
            for row in vol_values
        ]
        plot_implied_vols(
            strikes,
            filtered_vols,
            [labels[i] for i in indices],
            outfile=PLOT_IMPLIED_VOLS_FILE,
        )

    if failures:
        print("")
        print("Distributions with calibration or pricing issues:")
        for label, message in failures.items():
            print(f"  {label}: {message}")

    if vol_stats_df is not None:
        print("")
        print("Implied volatility stats across strikes")
        iv_header = (
            f"{'dist':<{label_width}}"
            f"{'mean':>12}{'std':>12}{'min':>12}{'max':>12}{'curv':>12}"
        )
        print(iv_header)
        print("-" * len(iv_header))
        for _, row in vol_stats_df.iterrows():
            print(
                f"{row['dist']:<{label_width}}"
                f"{row['mean']:12.6f}"
                f"{row['std']:12.6f}"
                f"{row['min']:12.6f}"
                f"{row['max']:12.6f}"
                f"{row['curvature_energy']:12.6f}"
            )


if __name__ == "__main__":
    main()