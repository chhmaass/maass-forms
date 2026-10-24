# Quality audit — 2026-10-24

Scope: all tracked source and documentation at baseline `c7d0944`, plus the proposed changes. Dimensions below are a qualitative engineering assessment, not a statistically estimated multivariate score. No single numeric grade can establish scientific correctness.

| Dimension | Baseline finding | Change and remaining limit |
|---|---|---|
| Mathematical correctness | Both equivalent points were pulled back before subtraction, making constraints numerically tautological | Raw-point versus pullback collocation; regression test rejects a zero/degenerate matrix. Still a truncated experiment, not a certified Hejhal solver |
| Numerical stability | Raw minimum singular values confounded spectral fit with Bessel decay and column scales | Column normalization, singular-value ratio, inverse scaling of coefficients; double-precision conditioning remains |
| Validation | No tests or independent residuals | Eight automated tests including off-spectrum comparison and held-out automorphy; no proof of spectral completeness |
| Search reliability | Refinement could leave requested bounds and lose the best coarse point | Bounded golden-section refinement retains coarse result; wide scans can still miss modes |
| Precision integrity | Cache rounded arguments to 10 decimal digits; refinement converted r to float despite elevated mpmath precision | Exact mpmath argument/precision cache keys; explicit double-precision SVD disclosure; no arbitrary-precision solve claim |
| Performance/resources | Unbounded cache; scalar rendering with repeated special functions | Bounded LRU; separated grid evaluation uses O(N × ny) Bessel evaluations instead of O(N × nx × ny). Wall-clock speedup not benchmarked |
| Portability/usability | Hard-coded Windows output path, GUI popup, ineffective parity switches | CLI for parity, search, modes, precision and output; JSON diagnostics; headless rendering |
| Reproducibility | Dependencies only described in prose; no CI | Installable project, dependency ranges, fixed seeds, JSON configuration, Python CI matrix. No exact dependency lock; cross-platform CI not yet verified |
| Defensive behavior | Invalid parity silently became odd; failed pullback silently returned; zero plots divided by zero | Explicit errors, finite matrix checks, input guards, restored precision context |
| Security | No credentials or network operations required by solver; numerical resource exhaustion possible on large inputs | No new runtime services; bounded cache; read-only CI token. No formal security or dependency-vulnerability audit performed |
| Maintainability | Monolithic experimental script and unused switches/comments | Removed misleading switches; retained importable helpers, package entry point, tests and documented contracts. A single module remains appropriate at this size |
| Scientific communication | “High precision refinement” and historical images overstated evidence | Explicit candidate status, independent residuals, historical-image warning and limitations |
| Licensing | README claimed MIT without a license file | Missing grant documented; author must choose and supply license text |

## Verification

- Local `MPLBACKEND=Agg python -m pytest -q`: eight tests pass.
- `python -m compileall -q maass_pipeline.py tests` and `git diff --check`: pass.
- Editable package installation and end-to-end CLI execution pass; PNG/JSON generation at 48×64 was verified, including visual inspection.
- A six-mode, 24-sample run (12 coarse points, 20 refinement steps) returned r=9.53416804665 and held-out relative error 0.0030464. Its difference from the reference is evidence of truncation error; these are not certified digits.
- Regression fixture near odd `r ≈ 9.533695`: six-mode fit must beat `r=9.3` by at least a factor of ten and have held-out relative error below 0.02. This is a useful falsification check, not a convergence theorem.

## Research priorities

1. Establish convergence under changes of truncation, sample height/density and spectral resolution, on both parity classes and several independently sourced reference modes.
2. Introduce truly arbitrary-precision linear algebra, conditioning diagnostics and controlled truncation/error bounds before claiming high precision or certification.
3. Enumerate and refine multiple minima with independent validation instead of returning one candidate per interval.
4. Clarify licensing and regenerate historical illustrations from validated settings.

Reference for the approximate odd spectral parameter: the computational overview [Maass forms and their L-functions](https://citeseerx.ist.psu.edu/document?doi=b6c3d596ff60836abdc7c593500767fd60172649&repid=rep1&type=pdf), which identifies odd level-one oldforms near 9.533695 and 12.173008. The test uses only approximate agreement and independent residuals, not a claim that every displayed decimal is verified.
