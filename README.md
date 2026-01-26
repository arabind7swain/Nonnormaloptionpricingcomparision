# Distribution Properties

The table below summarizes key characteristics for each log-return distribution
implemented in this project.

| Distribution | # Params | Symmetric? | Skew Control | Excess Kurtosis / Tails |
|--------------|---------:|------------|--------------|------------------|
| Normal | 2 (μ, σ) | Yes | None | 0 (light tails) |
| Logistic | 2 (μ, scale) | Yes | None | 1.2 (sub-exponential) |
| Laplace | 2 (μ, scale) | Yes | None | 3.0 (double-exponential) |
| Hyperbolic secant | 2 (μ, scale) | Yes | None | 2.0 |
| Champernowne | 3 (μ, scale, d) | Yes | None | Varies with d (heavy tails for d<0) |
| GED | 3 (μ, scale, p) | Yes | None | From 3 (p=1) to 0 (p=2) |
| Johnson SU | 4 (a, b, loc, scale) | No | 1 skew (a) | Wide range incl. heavy tails (excess varies) |
| Tukey lambda | 3 (loc, scale, λ) | Yes | None | Tail varies; finite moments depend on λ |
| Hyperbolic | 4 (α, β, δ, μ) | No | β | Heavy but finite tails (excess > 0) |
| Generalized hyperbolic | 5 (λ, α, β, δ, μ) | No | β | Controls both skew/tails widely |
| Variance gamma | 4 (μ, θ, σ, ν) | No | θ | Potentially infinite excess (depends on ν) |
| Normal-inverse Gaussian | 4 (α, β, δ, μ) | No | β | Potentially infinite excess (depends on α,β) |
| CGMY | 6 (C, G, M, Y, scale, μ) | No | G vs M | Tempered power-law tails (finite moments if Y < order) |
| NEF-GHS | 4 (μ, scale, κ, θ) | No | θ | Heavy, log-sech-like tails |
| GSH | 3 (μ, scale, t) | Yes | None | Tails controlled by t (excess varies) |
| SGSH | 4 (μ, scale, t, skew) | No | skew | Same as GSH |
| Skew normal | 3 (shape, loc, scale) | No | shape | Excess kurtosis near 0 |
| Skew-t (Jones–Faddy) | 4 (a, b, loc, scale) | No | (a−b) | Heavy tails; excess depends on a,b |
| Asymmetric Laplace | 3 (κ, loc, scale) | No | κ | Heavy exponential tails (excess 3 regardless of κ) |
| Noncentral t | 3 (df, nc, scale) | No | nc | Heavy algebraic tails (excess finite for df>4) |
