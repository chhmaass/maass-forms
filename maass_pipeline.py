from mpmath import mp
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# ============================================================
#  Maass cusp forms for SL(2,Z) — EVEN + ODD (FAST SCAN -> REFINE + CHEAP RENDER)
#  Hejhal-style truncated reconstruction + visualization
#
#  PERFORMANCE UPDATES:
#   ✅ Cache expensive Bessel-K evaluations (dominant bottleneck)
#   ✅ Deterministic y-grid during FAST SCAN for maximal cache hits
#   ✅ FAST SCAN small; high precision only in REFINE
#   ✅ Render uses lower mp.dps + lower resolution
#
#  VISUALIZATION FIX:
#   ✅ Use the *action band* in y: y in [0.65, 1.45]
#   ✅ Stronger quantile clipping to avoid near-zero desert dominating
#
#  CONCEPTUAL NOTE:
#   This pipeline numerically reconstructs weight-0 Maass cusp forms
#   (principal series, K-type 0) as Laplacian eigenfunctions on
#   SL(2,Z)\H, following the modern numerical spectral program.
#
#  Run: python maass_pipeline.py
# ============================================================

# ----------------------------
# Switches (edit these)
# ----------------------------
RUN_EVEN = True
RUN_ODD = False
COMBINE_EVEN_ODD = True
VERBOSE = True

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
RENDER_CLIP_Q = (0.15, 0.85)


def log(msg: str) -> None:
    if VERBOSE:
        print(msg, flush=True)


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
    w = z
    for _ in range(max_steps):
        w = reduce_T(w)
        if abs(w) < 1:
            w = S(w)
            continue
        if in_fundamental_domain(w):
            return w
    return w


# ============================================================
# Cached Bessel-K core
# ============================================================
_BESSEL_CACHE = {}


def _cache_key(r, n, y, phase_tag, r_digits=10, y_digits=10):
    return (
        phase_tag,
        mp.dps,
        round(float(r), r_digits),
        int(n),
        round(float(y), y_digits),
    )


def K_ir(r, n, y, phase_tag="scan"):
    key = _cache_key(r, n, y, phase_tag)
    val = _BESSEL_CACHE.get(key)
    if val is None:
        val = mp.besselk(1j * r, 2 * mp.pi * n * y)
        _BESSEL_CACHE[key] = val
    return val


def clear_bessel_cache():
    _BESSEL_CACHE.clear()


# ----------------------------
# Maass basis
# ----------------------------
def maass_basis(z, r, N, parity="even", phase_tag="scan"):
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
        z1 = pullback(z)
        z2 = pullback(S(z))
        v1 = maass_basis(z1, r, N, parity, phase_tag)
        v2 = maass_basis(z2, r, N, parity, phase_tag)
        rows.append([complex(v1[i] - v2[i]) for i in range(N)])
    return np.array(rows, dtype=np.complex128)


def smallest_singular_value(samples, r, N, parity="even", phase_tag="scan"):
    M = build_matrix(samples, float(r), N, parity, phase_tag)
    return float(np.linalg.svd(M, compute_uv=False)[-1])


def extract_coeffs(samples, r, N, parity="even", phase_tag="refine"):
    M = build_matrix(samples, float(r), N, parity, phase_tag)
    _, Svals, Vh = np.linalg.svd(M)
    A = Vh.conj().T[:, -1]
    k = int(np.argmax(np.abs(A)))
    if abs(A[k]) > 1e-300:
        A = A / A[k]
    return float(Svals[-1]), A


# ----------------------------
# Sampling
# ----------------------------
def sample_points_random(m=140, seed=0, y_min=0.75, y_max=1.45):
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(m):
        x = rng.uniform(-0.5, 0.5)
        y = rng.uniform(y_min, y_max)
        pts.append(mp.mpf(x) + 1j * mp.mpf(y))
    return pts


def sample_points_grid(
    m=90, seed=0, x_min=-0.5, x_max=0.5, y_min=0.80, y_max=1.40
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
    r_fine = np.linspace(r0 - refine_width, r0 + refine_width, refine_n)
    sig_fine = np.empty_like(r_fine)

    for i, r in enumerate(r_fine):
        sig_fine[i] = smallest_singular_value(samples, r, N, parity, phase_tag)

    return float(r_fine[int(np.argmin(sig_fine))]), float(sig_fine.min())


# ----------------------------
# Maass form evaluation
# ----------------------------
def make_f(r, A, parity="even", phase_tag="eval"):
    def f(z):
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

    return f


# ============================================================
# Render
# ============================================================
def render_maass(f, title):
    mp.dps = RENDER_DPS
    xs = np.linspace(RENDER_X_MIN, RENDER_X_MAX, RENDER_NXNY[0])
    ys = np.linspace(RENDER_Y_MIN, RENDER_Y_MAX, RENDER_NXNY[1])
    Z = np.zeros((len(ys), len(xs)))

    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            Z[i, j] = float(mp.re(f(pullback(mp.mpf(x) + 1j * mp.mpf(y)))))

    a = np.quantile(np.abs(Z), 0.85)
    Z = np.clip(Z, -a, a) / a
    Z = np.sign(Z) * np.abs(Z) ** RENDER_GAMMA

    plt.figure(figsize=(6.4, 9.6))
    plt.contourf(xs, ys, Z, levels=RENDER_LEVELS, cmap="RdBu_r")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()

    os.makedirs(r"C:\Maass", exist_ok=True)
    plt.savefig(r"C:\Maass\Figure_MaassForms.png", dpi=300)
    plt.show()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    mp.dps = 35
    clear_bessel_cache()
    samples = sample_points_grid(m=60, seed=1)

    r_even, _ = find_r(samples, 16, parity="even", r_lo=18, r_hi=35)

    mp.dps = 70
    clear_bessel_cache()
    samples_ref = sample_points_random(m=160, seed=2)
    _, A_even = extract_coeffs(samples_ref, r_even, 40, parity="even")

    f_even = make_f(r_even, A_even, parity="even")
    render_maass(
        f_even,
        title="Even Maass cusp form for $SL(2,\\mathbb{Z})$ — spectral geometry after Hans Maass",
    )
