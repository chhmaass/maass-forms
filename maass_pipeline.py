"""Experimental Fourier–Bessel collocation; candidates are not certified forms."""
from mpmath import mp
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
import json
from pathlib import Path
from functools import lru_cache

# ----------------------------
# Cheap render config
# ----------------------------
RENDER_NXNY = (320, 420)
RENDER_DPS = 25
RENDER_Y_MIN = 0.75
RENDER_Y_MAX = 1.35
RENDER_X_MIN = -0.25
RENDER_X_MAX = 0.25
RENDER_LEVELS = 16
RENDER_GAMMA = 0.7


# ----------------------------
# Fundamental domain utilities
# ----------------------------
def reduce_T(z):
    k = mp.floor(mp.re(z) + mp.mpf("0.5"))
    return z - k


def S(z):
    return -1 / z


def in_fundamental_domain(z):
    return (
        mp.im(z) > 0
        and abs(mp.re(z)) <= mp.mpf("0.5")
        and abs(z) >= mp.mpf("1.0")
    )


def pullback(z, max_steps=30):
    if not mp.isfinite(mp.re(z)) or not mp.isfinite(mp.im(z)) or mp.im(z) <= 0:
        raise ValueError("z must be finite and in the upper half-plane")
    w = z
    for _ in range(max_steps):
        w = reduce_T(w)
        if abs(w) < 1:
            w = S(w)
            continue
        if in_fundamental_domain(w):
            return w
    raise RuntimeError("fundamental-domain reduction did not converge")


# ============================================================
# Cached Bessel-K core
# ============================================================
@lru_cache(maxsize=32768)
def _cached_bessel(r, n, y, dps):
    with mp.workdps(dps):
        return mp.re(mp.besselk(mp.j * r, 2 * mp.pi * n * y))


def K_ir(r, n, y, phase_tag="scan"):
    r, y = mp.mpf(r), mp.mpf(y)
    if not mp.isfinite(r) or not mp.isfinite(y) or y <= 0 or n < 1:
        raise ValueError("invalid Bessel arguments")
    return _cached_bessel(r, n, y, mp.dps)


def clear_bessel_cache():
    _cached_bessel.cache_clear()


def validate_basis(z, N, parity):
    if parity not in ("even", "odd"):
        raise ValueError("parity must be even or odd")
    if not isinstance(N, int) or N < 1:
        raise ValueError("N must be a positive integer")
    if not mp.isfinite(mp.re(z)) or not mp.isfinite(mp.im(z)) or mp.im(z) <= 0:
        raise ValueError("z must be finite and in the upper half-plane")


# ----------------------------
# Maass basis
# ----------------------------
def maass_basis(z, r, N, parity="even", phase_tag="scan"):
    validate_basis(z, N, parity)
    x, y = mp.re(z), mp.im(z)
    sqy = mp.sqrt(y)
    vals = []
    for n in range(1, N + 1):
        K = K_ir(r, n, y, phase_tag)
        trig = (
            mp.cos(2 * mp.pi * n * x)
            if parity == "even"
            else mp.sin(2 * mp.pi * n * x)
        )
        vals.append(sqy * K * trig)
    return vals


# ----------------------------
# Automorphy constraint
# ----------------------------
def build_matrix(samples, r, N, parity="even", phase_tag="scan"):
    rows = []
    for z in samples:
        z1 = z
        z2 = pullback(z)
        v1 = maass_basis(z1, r, N, parity, phase_tag)
        v2 = maass_basis(z2, r, N, parity, phase_tag)
        rows.append([complex(v1[i] - v2[i]) for i in range(N)])
    M = np.array(rows, dtype=np.complex128)
    if M.ndim != 2 or M.shape[0] < N:
        raise ValueError("at least N samples are required")
    if not np.all(np.isfinite(M)):
        raise ValueError("non-finite collocation matrix")
    return M


def scaled_matrix(samples, r, N, parity, phase_tag):
    M = build_matrix(samples, r, N, parity, phase_tag)
    scales = np.linalg.norm(M, axis=0)
    if np.any(scales <= np.finfo(float).tiny):
        raise ValueError("degenerate constraints: use points below the unit circle or fewer modes")
    return M / scales, scales


def smallest_singular_value(samples, r, N, parity="even", phase_tag="scan"):
    M, _ = scaled_matrix(samples, r, N, parity, phase_tag)
    values = np.linalg.svd(M, compute_uv=False)
    return float(values[-1] / values[0])


def extract_coeffs(samples, r, N, parity="even", phase_tag="refine"):
    M, scales = scaled_matrix(samples, r, N, parity, phase_tag)
    _, Svals, Vh = np.linalg.svd(M, full_matrices=False)
    A = Vh.conj().T[:, -1] / scales
    k = int(np.argmax(np.abs(A)))
    if abs(A[k]) > 1e-300:
        A = A / A[k]
    return float(Svals[-1] / Svals[0]), A


# ----------------------------
# Sampling
# ----------------------------
def sample_points_random(m=140, seed=0, y_min=0.35, y_max=0.55):
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(m):
        x = rng.uniform(-0.5, 0.5)
        y = rng.uniform(y_min, y_max)
        pts.append(mp.mpf(x) + 1j * mp.mpf(y))
    return pts


def sample_points_grid(
    m=90, seed=0, x_min=-0.5, x_max=0.5, y_min=0.35, y_max=0.55
):
    rng = np.random.default_rng(seed)
    y_grid = [mp.mpf(v) for v in np.linspace(
        y_min, y_max, max(10, int(np.sqrt(m))))]
    x_grid = np.linspace(x_min, x_max, max(10, int(np.sqrt(m))))
    pts = []
    for i in range(m):
        x = float(rng.choice(x_grid))
        y = y_grid[i % len(y_grid)]
        pts.append(mp.mpf(x) + 1j * y)
    return pts


# ----------------------------
# Spectral scan
# ----------------------------
def find_r(
    samples,
    N,
    parity="even",
    r_lo=7.0,
    r_hi=18.0,
    n_grid=80,
    refine_width=0.35,
    refine_n=80,
    phase_tag="scan",
    progress_every=20,
):
    if not (0 <= r_lo < r_hi) or n_grid < 3 or refine_n < 3 or refine_width <= 0:
        raise ValueError("invalid spectral search bounds or resolution")
    r_grid = np.linspace(r_lo, r_hi, n_grid)
    sig = np.empty_like(r_grid)

    t0 = time.time()
    for i, r in enumerate(r_grid):
        sig[i] = smallest_singular_value(samples, r, N, parity, phase_tag)
        if progress_every and (i + 1) % progress_every == 0:
            dt = time.time() - t0
            print(
                f"  scan {i+1}/{n_grid} elapsed={dt:.1f}s  best≈{sig[:i+1].min():.3e}",
                flush=True,
            )

    r0 = float(r_grid[int(np.argmin(sig))])
    refine_width = min(refine_width, (r_hi - r_lo) / (n_grid - 1))
    lo, hi = max(r_lo, r0 - refine_width), min(r_hi, r0 + refine_width)
    # Golden-section minimization inside the best coarse cell; retain the
    # coarse result so boundary candidates cannot be silently discarded.
    ratio = (np.sqrt(5) - 1) / 2
    left, right = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    objective = lambda r: smallest_singular_value(samples, r, N, parity, phase_tag)
    fl, fr = objective(left), objective(right)
    for _ in range(refine_n):
        if fl < fr:
            hi, right, fr = right, left, fl
            left = hi - ratio * (hi - lo)
            fl = objective(left)
        else:
            lo, left, fl = left, right, fr
            right = lo + ratio * (hi - lo)
            fr = objective(right)
    candidates = [(r0, float(sig.min())), (float(left), fl), (float(right), fr)]
    return min(candidates, key=lambda item: item[1])



# ----------------------------
# Maass form evaluation
# ----------------------------
def make_f(r, A, parity="even", phase_tag="eval"):
    A = tuple(A)
    validate_basis(mp.j, len(A), parity)
    def f(z):
        validate_basis(z, len(A), parity)
        x, y = mp.re(z), mp.im(z)
        sqy = mp.sqrt(y)
        val = mp.mpf("0")
        for n, An in enumerate(A, start=1):
            K = K_ir(r, n, y, phase_tag)
            trig = (
                mp.cos(2 * mp.pi * n * x)
                if parity == "even"
                else mp.sin(2 * mp.pi * n * x)
            )
            val += An * sqy * K * trig
        return val

    def evaluate_grid(xs, ys):
        # Separation of variables: only len(ys)*N Bessel calls.
        n = np.arange(1, len(A) + 1)
        angles = 2 * np.pi * n[:, None] * np.asarray(xs)[None, :]
        trig = np.cos(angles) if parity == "even" else np.sin(angles)
        radial = np.array([[complex(a * mp.sqrt(mp.mpf(float(y))) *
                           K_ir(r, k, mp.mpf(float(y)), phase_tag))
                            for k, a in enumerate(A, 1)] for y in ys])
        return np.real(radial @ trig)
    f.evaluate_grid = evaluate_grid
    return f


# ============================================================
# Render
# ============================================================
def render_maass(f, title, output="Figure_MaassForms.png", show=False):
    with mp.workdps(RENDER_DPS):
        return _render_maass(f, title, output, show)


def _render_maass(f, title, output, show):
    xs = np.linspace(RENDER_X_MIN, RENDER_X_MAX, RENDER_NXNY[0])
    ys = np.linspace(RENDER_Y_MIN, RENDER_Y_MAX, RENDER_NXNY[1])
    Z = np.zeros((len(ys), len(xs)))

    if hasattr(f, "evaluate_grid"):
        Z = f.evaluate_grid(xs, ys)
    else:
        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                Z[i, j] = float(mp.re(f(mp.mpc(float(x), float(y)))))

    a = np.quantile(np.abs(Z), 0.85)
    if not np.isfinite(a) or a <= np.finfo(float).tiny:
        raise ValueError("cannot render a zero or non-finite field")
    Z = np.clip(Z, -a, a) / a
    Z = np.sign(Z) * np.abs(Z) ** RENDER_GAMMA

    plt.figure(figsize=(6.4, 9.6))
    plt.contourf(xs, ys, Z, levels=RENDER_LEVELS, cmap="RdBu_r")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150)
    if show:
        plt.show()
    plt.close()
    return output


def relative_automorphy_error(f, samples):
    """Held-out error on raw points, without making invariance tautological."""
    left = np.array([complex(f(z)) for z in samples])
    right = np.array([complex(f(pullback(z))) for z in samples])
    scale = np.linalg.norm(left) + np.linalg.norm(right)
    if scale <= np.finfo(float).tiny:
        raise ValueError("cannot validate a zero field")
    return float(2 * np.linalg.norm(left - right) / scale)


# ============================================================
# MAIN
# ============================================================
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parity", choices=("even", "odd"), default="odd")
    parser.add_argument("--r-lo", type=float, default=9.3)
    parser.add_argument("--r-hi", type=float, default=9.8)
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--grid", type=int, default=20)
    parser.add_argument("--refine-steps", type=int, default=24)
    parser.add_argument("--dps", type=int, default=30)
    parser.add_argument("--output", default="output/maass_candidate.png")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args(argv)
    if args.dps < 15 or args.modes < 1 or args.samples < args.modes:
        parser.error("require dps >= 15 and samples >= modes >= 1")
    with mp.workdps(args.dps):
        samples = sample_points_grid(m=args.samples, seed=1)
        r, score = find_r(samples, args.modes, args.parity, args.r_lo, args.r_hi,
                          args.grid, refine_n=args.refine_steps, progress_every=0)
        _, coefficients = extract_coeffs(samples, r, args.modes, args.parity)
        f = make_f(r, coefficients, args.parity)
        error = relative_automorphy_error(f, sample_points_random(m=args.samples, seed=42))
        report = dict(parity=args.parity, r=r, eigenvalue=0.25+r*r,
                      singular_ratio=score, held_out_relative_error=error,
                      modes=args.modes, samples=args.samples, dps=args.dps,
                      svd_dtype="complex128", certified=False,
                      search_bounds=[args.r_lo, args.r_hi], grid=args.grid,
                      refine_steps=args.refine_steps, fitting_seed=1, validation_seed=42,
                      coefficients=[[float(v.real), float(v.imag)] for v in coefficients])
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.with_suffix(".json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report, indent=2))
        if not args.no_render:
            render_maass(f, f"{args.parity.title()} candidate, r={r:.6f} (uncertified)", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
