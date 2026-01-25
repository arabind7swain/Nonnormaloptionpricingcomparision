"""Simulate mixture data and fit several candidate distributions."""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy import stats

from fit_distribution import (
    fit_distribution,
    plot_densities,
    simulate_mixture,
)

DENSITY_PLOT_FILE: Optional[str] = "fit_densities.png"


def main() -> None:
    """Simulate mixture draws and compare fitted distributions."""
    nobs = 1000
    print("#obs:", nobs)
    components = [
        (0.1, -0.3, 3.0),
        (0.9, 0.3, 1.0),
    ]
    print("Mixture components (weight, mean, std):")
    for w, m, s in components:
        print(f"  w={w:.3f}, mean={m:.3f}, std={s:.3f}")
    data = simulate_mixture(nobs, components, seed=12345)
    descriptives = {
        "mean": float(np.mean(data)),
        "std": float(np.std(data, ddof=1)),
        "skew": float(stats.skew(data)),
        "kurtosis": float(stats.kurtosis(data, fisher=False)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
    }
    print(
        f"Descriptive stats (n={data.size}): "
        + ", ".join(f"{key}={value:.4f}" for key, value in descriptives.items())
    )
    candidates = ["normal", "skew_normal", "gsh", "sgsh", "t", "nct"]
    records: list[dict[str, float | str]] = []
    pdf_entries: list[tuple[str, Callable[[np.ndarray], np.ndarray]]] = []
    total_start = time.perf_counter()
    for name in candidates:
        try:
            fit, pdf_func = fit_distribution(data, name)
            records.append(fit)
            pdf_entries.append((name, pdf_func))
            params_fmt = ", ".join(f"{p:.4f}" for p in fit["params"])
            print(
                f"{name:>8s} loglike={fit['loglike']:.2f} "
                f"ks={fit['ks_stat']:.3f} p={fit['ks_pvalue']:.3f} "
                f"params=({params_fmt}) "
                f"n={fit['n_params']} time={fit['fit_time']:.3f}s"
            )
        except Exception as exc:  # pragma: no cover - diagnostics for unsupported fits.
            print(f"{name:>8s} fit failed: {exc}")
    if records:
        if DENSITY_PLOT_FILE is not None:
            plot_densities(data, components, pdf_entries, DENSITY_PLOT_FILE)
        df = pd.DataFrame(records)
        column_order = [
            "distribution",
            "scipy_name",
            "loglike",
            "mean_density",
            "std_density",
            "skew_density",
            "kurt_density",
            "ks_stat",
            "ks_pvalue",
            "n_params",
            "fit_time",
            "params",
            "param_names",
        ]
        df = df[[col for col in column_order if col in df.columns]]
        df.to_csv("fit_summary.csv", index=False)
        print("Fit summary saved to fit_summary.csv")
    total_elapsed = time.perf_counter() - total_start
    print(f"Total fitting time: {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()