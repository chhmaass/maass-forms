import numpy as np
import pytest
from mpmath import mp
import maass_pipeline as p


def test_pullback_and_invalid_domain():
    for z in (mp.mpc('.2', '.4'), mp.mpc('3.1', '.2')):
        assert p.in_fundamental_domain(p.pullback(z))
        assert abs(p.pullback(z) - p.pullback(p.S(z))) < mp.mpf('1e-12')
    with pytest.raises(ValueError):
        p.pullback(0)
    with pytest.raises(RuntimeError):
        p.pullback(mp.mpc('.2', '.4'), max_steps=0)


def test_constraints_are_not_tautological():
    samples = p.sample_points_grid(m=12)
    matrix = p.build_matrix(samples, 9.5, 3, 'odd')
    assert np.linalg.norm(matrix) > 1e-9
    with pytest.raises(ValueError, match='degenerate'):
        p.smallest_singular_value([mp.mpc('.1', 2)] * 4, 9.5, 3)


def test_exact_bounded_precision_aware_cache():
    p.clear_bessel_cache()
    with mp.workdps(35):
        a = p.K_ir(mp.mpf('9.5'), 1, mp.mpf('.4'))
        b = p.K_ir(mp.mpf('9.500000000001'), 1, mp.mpf('.4'))
        assert a != b
        assert abs(a-mp.re(mp.besselk(mp.j*mp.mpf('9.5'), 2*mp.pi*mp.mpf('.4')))) < mp.mpf('1e-35')
    with mp.workdps(20):
        p.K_ir(9.5, 1, mp.mpf('.4'))
    assert p._cached_bessel.cache_info().currsize == 3
    assert p._cached_bessel.cache_info().maxsize == 32768


def test_basis_symmetries_and_validation():
    for parity, sign in [('even', 1), ('odd', -1)]:
        f = p.make_f(9.5, [1, .2], parity)
        z = mp.mpc('.17', '.4')
        assert abs(f(z+1)-f(z)) < 1e-15
        assert abs(f(-mp.conj(z))-sign*f(z)) < 1e-15
    with pytest.raises(ValueError):
        p.maass_basis(mp.j, 9.5, 2, 'typo')
    with pytest.raises(ValueError):
        p.find_r([], 2, r_lo=5, r_hi=4)


def test_known_odd_mode_beats_off_spectrum_and_generalizes():
    samples = p.sample_points_grid(m=24, seed=1)
    with mp.workdps(25):
        # First odd PSL(2,Z) cusp eigenparameter, independently tabulated.
        r = 9.533695
        score, coefficients = p.extract_coeffs(samples, r, 6, 'odd')
        assert score < p.smallest_singular_value(samples, 9.3, 6, 'odd') / 10
        f = p.make_f(r, coefficients, 'odd')
        assert p.relative_automorphy_error(f, p.sample_points_random(m=18, seed=42)) < .02


def test_render_zero_and_precision_restored(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'RENDER_NXNY', (8, 8))
    old = mp.dps
    with pytest.raises(ValueError, match='zero'):
        p.render_maass(lambda z: 0, 'zero', tmp_path/'zero.png')
    assert mp.dps == old
    out = p.render_maass(lambda z: mp.re(z), 'test', tmp_path/'test.png')
    assert out.stat().st_size > 0
    assert mp.dps == old


def test_separated_render_matches_scalar():
    f = p.make_f(9.5, [1, .2], 'odd')
    xs, ys = [-.2, .1], [.8, 1.1]
    expected = [[float(mp.re(f(mp.mpc(x, y)))) for x in xs] for y in ys]
    np.testing.assert_allclose(f.evaluate_grid(xs, ys), expected, rtol=1e-12, atol=1e-20)


def test_search_refines_interior_and_keeps_boundary(monkeypatch):
    monkeypatch.setattr(p, 'smallest_singular_value',
                        lambda samples, r, *args: (r - 1.2345) ** 2)
    r, score = p.find_r([], 1, r_lo=1, r_hi=2, n_grid=6,
                        refine_n=24, progress_every=0)
    assert abs(r - 1.2345) < 1e-5
    assert score < 1e-10
    monkeypatch.setattr(p, 'smallest_singular_value',
                        lambda samples, r, *args: r)
    r, _ = p.find_r([], 1, r_lo=1, r_hi=2, n_grid=6,
                    refine_n=5, progress_every=0)
    assert r == 1
