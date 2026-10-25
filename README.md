# Maass Forms — experimental spectral geometry

An educational Fourier–Bessel collocation experiment for weight-zero Maass cusp forms on the modular surface. **Outputs are numerical candidates, not certified eigenforms.**

## Mathematical model

For spectral parameter `r`, the truncated basis is

- even: `sqrt(y) K_(ir)(2 pi n y) cos(2 pi n x)`;
- odd: `sqrt(y) K_(ir)(2 pi n y) sin(2 pi n x)`.

These functions satisfy translation symmetry, parity, cusp decay and the hyperbolic Laplace eigen-equation with eigenvalue `1/4 + r²`. Modular invariance remains a numerical constraint: compare the expansion at **raw sample z** with its fundamental-domain pullback. Reducing both sides first makes this constraint tautological and must be avoided.

Samples lie below the unit circle (default heights 0.35–0.55), where pullback produces nontrivial constraints. Columns are normalized before double-precision SVD; the search minimizes the ratio of smallest to largest singular value. Coefficients are transformed back to the original basis. An independent random sample measures relative automorphy error as `2 ||f(z)-f(pullback(z))|| / (||f(z)||+||f(pullback(z))||)`.

## Install and run

Python 3.10+:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install '.[test]'
python maass_pipeline.py --help
python maass_pipeline.py --no-render
python maass_pipeline.py --output output/odd.png
python -m pytest -q
```

Default search: odd parity, `r=9.3–9.8`, 8 modes, 32 fitting points, 20 coarse points and 24 golden-section refinement steps, 30 decimal digits for Bessel evaluation. Select `--parity even --r-lo 13.5 --r-hi 14` for a different experiment. Both parity choices are implemented; each run fits one parity. Search and rendering may take time.

Refinement stays inside the best coarse cell and retains endpoint candidates.

The CLI writes a JSON sidecar with the candidate parameter, eigenvalue, coefficients, normalized singular ratio, held-out error and numerical settings. Images use a portable configurable path, work without a GUI (`MPLBACKEND=Agg`), and do not open a window by default. Separated radial/angular evaluation reuses each Bessel value across a row. The displayed field is the raw truncated expansion, without imposing invariance through pullback. Quantile clipping and gamma adjustment emphasize structure; image colors are not quantitative amplitudes.

## Reliability and limitations

- Bessel evaluation uses mpmath; SVD is NumPy `complex128`. Raising `--dps` does **not** make the linear solve arbitrary precision.
- Cache entries preserve exact mpmath arguments and precision, and are bounded to 32,768 entries. Precision changes do not reuse low-precision values.
- A small fitting singular ratio alone is insufficient. Check held-out error and repeat with more modes, points, precision and denser search grids. Compare candidate stability. The CLI reports diagnostics but does not automatically accept a form.
- A coarse global minimum can miss eigenvalues, choose an endpoint, or identify a truncation artifact. This is not a complete implementation of Hejhal's algorithm, a validated eigensolver, or a proof of existence.
- Truncation at low heights and double-precision conditioning limit accuracy. Full arbitrary-precision linear algebra, adaptive spectral root finding and truncation error bounds remain research work.
- The checked-in historical PNGs were produced before the constraint correction. They are illustrations, **not validation of the corrected solver**.

Tests cover nontrivial constraints, pullback, cache precision, parity/periodicity, rejection of degenerate inputs, zero-field rendering and independent residuals near an odd spectral candidate. CI runs on Python 3.10, 3.12 and 3.13. No network access, credentials or external services are needed by the solver.

See [QUALITY.md](QUALITY.md) for the audit and its verification limits.

## Author and licensing

Christoph H. Maass. The previous README stated “MIT License”, but no license text or copyright grant was committed. Licensing should be clarified by the author before redistribution; this update does not invent a legal grant.
