"""
Tests for beam/generation.py: make_bunch (6D covariance, Cholesky) and the make_simple_bunch* wrappers.
"""

import os
import sys

import numpy as np
import pytest
from scipy.stats import kurtosis

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from pmd_beamphysics import ParticleGroup

import FACET2_S2E as qs
from FACET2_S2E.beam.generation import _covariance_sqrt


C = 3e8  # the package's convention for z = -c*t


def coords(P):
    """(x, xp, y, yp, z, pz) of a ParticleGroup, with z = -c*t."""
    return np.vstack((P.x, P.xp, P.y, P.yp, -C*P.t, P.pz))


@pytest.fixture
def source_beam_file(tmp_path):
    """A correlated beam on disk (x-xp tilt, z-pz chirp, x-pz dispersion), as make_simple_bunch reads it."""
    rng = np.random.default_rng(0)
    n = 40000
    t = rng.normal(2e-12, 300e-15, n)
    z = -C*(t - t.mean())
    pz = 125e6*(1 + 2e-3*z/np.std(z) + 1e-3*rng.normal(size=n))
    x = rng.normal(0, 100e-6, n) + 0.05*(pz/125e6 - 1)
    xp = -0.6*x/100e-6*20e-6 + 0.8*rng.normal(0, 20e-6, n)
    y = rng.normal(50e-6, 80e-6, n)
    yp = rng.normal(0, 15e-6, n)
    P = ParticleGroup(data=dict(x=x, px=xp*pz, y=y, py=yp*pz, z=np.zeros(n), t=t, pz=pz,
                                status=np.ones(n, dtype=int), weight=np.full(n, 1.6e-9/n), species='electron'))
    base = str(tmp_path / "source")
    P.write(base + ".h5")
    return base, P


class TestCovarianceSqrt:
    def test_reproduces_full_rank_covariance(self):
        rng = np.random.default_rng(1)
        scales = np.array([1e-4, 1e-5, 1e-4, 1e-5, 1e-4, 1e5])  # mixed units, as in a real beam
        R = np.corrcoef(rng.normal(size=(6, 50)))
        sigma = R*np.outer(scales, scales)
        A = _covariance_sqrt(sigma)
        assert np.allclose(A @ A.T, sigma, rtol=1e-9, atol=0)
        assert np.allclose(A, np.tril(A))  # Cholesky factor

    def test_zero_variance_coordinates_are_left_out(self):
        sigma = np.diag([1e-8, 0, 4e-8, 0, 1e-8, 0.0])
        sigma[0, 4] = sigma[4, 0] = 0.5e-8
        A = _covariance_sqrt(sigma)
        assert np.allclose(A @ A.T, sigma, rtol=1e-12, atol=0)
        assert np.all(A[[1, 3, 5], :] == 0) and np.all(A[:, [1, 3, 5]] == 0)

    def test_rank_deficient_falls_back_to_psd_sqrt(self):
        # y = 2x exactly: not positive definite, Cholesky fails
        sigma = np.zeros((6, 6))
        sigma[0, 0], sigma[2, 2], sigma[0, 2], sigma[2, 0] = 1e-12, 4e-12, 2e-12, 2e-12
        A = _covariance_sqrt(sigma)
        assert np.allclose(A @ A.T, sigma, rtol=1e-6, atol=1e-24)

    def test_all_zero(self):
        assert np.all(_covariance_sqrt(np.zeros((6, 6))) == 0)


class TestMakeBunch:
    def test_moments_means_and_charge(self):
        np.random.seed(2)
        rms = np.array([1e-4, 2e-5, 3e-4, 4e-5, 5e-5, 1e5])
        means = [1e-3, 2e-4, -1e-3, -2e-4, 3e-3, 1e9]
        P = qs.make_bunch(100000, np.diag(rms**2), means, charge=2e-9)
        X = coords(P)
        assert len(P) == 100000
        assert P.charge == pytest.approx(2e-9)
        assert np.allclose(np.std(X, axis=1), rms, rtol=0.01)
        assert np.mean(P.x) == pytest.approx(1e-3, abs=1e-6)
        assert np.mean(P.xp) == pytest.approx(2e-4, abs=1e-6)
        assert np.mean(P.y) == pytest.approx(-1e-3, abs=3e-6)
        assert np.mean(P.yp) == pytest.approx(-2e-4, abs=1e-6)
        assert np.mean(P.t)*C == pytest.approx(3e-3, rel=1e-3)  # means[4] is c*<t>
        assert np.mean(P.pz) == pytest.approx(1e9, rel=1e-5)
        assert np.all(P.z == 0)

    def test_correlations_are_reproduced(self):
        np.random.seed(3)
        rms = np.array([1e-4, 2e-5, 1e-4, 2e-5, 1e-4, 1e5])
        R = np.eye(6)
        R[0, 1] = R[1, 0] = -0.7   # alpha > 0
        R[4, 5] = R[5, 4] = 0.5    # chirp, head (z > 0) has more momentum
        R[0, 5] = R[5, 0] = 0.3    # dispersion
        P = qs.make_bunch(200000, R*np.outer(rms, rms), [0, 0, 0, 0, 0, 1e9])
        Rgot = np.corrcoef(coords(P))
        assert np.allclose(Rgot, R, atol=0.01)
        # sign convention: head (positive z = earlier t) has more momentum
        assert np.corrcoef(P.t, P.pz)[0, 1] < -0.4

    def test_zero_moments(self):
        np.random.seed(4)
        P = qs.make_bunch(1000, np.diag([1e-8, 0, 0, 0, 1e-8, 0.0]), [0, 0, 5e-4, 1e-4, 0, 1e9])
        assert np.all(P.y == 5e-4)
        assert np.allclose(P.yp, 1e-4)
        assert np.all(P.pz == 1e9)

    def test_flat_profile(self):
        np.random.seed(5)
        P = qs.make_bunch(200000, np.diag([1e-8, 1e-10, 1e-8, 1e-10, 1e-8, 1e10]), [0, 0, 0, 0, 0, 1e9], t_profile='flat')
        assert np.std(P.t)*C == pytest.approx(1e-4, rel=0.01)
        # generalized normal of order 4: excess kurtosis Gamma(5/4)Gamma(1/4)/Gamma(3/4)^2 - 3 = -0.81
        assert kurtosis(P.t) == pytest.approx(-0.81, abs=0.03)
        assert abs(kurtosis(P.x)) < 0.05

    def test_invalid_profile(self):
        with pytest.raises(ValueError):
            qs.make_bunch(10, np.eye(6), [0]*6, t_profile='square')

    def test_save_path(self, tmp_path):
        qs.make_bunch(10, np.eye(6)*1e-12, [0]*5 + [1e9])
        assert not any(tmp_path.iterdir())
        P = qs.make_bunch(10, np.eye(6)*1e-12, [0]*5 + [1e9], save_path=str(tmp_path / "b"))
        Q = ParticleGroup(str(tmp_path / "b.h5"))
        assert np.allclose(Q.pz, P.pz)


class TestMakeSimpleBunch:
    def test_uncorrelated_matches_rms(self, source_beam_file, tmp_path):
        base, S = source_beam_file
        np.random.seed(6)
        P = qs.make_simple_bunch(base, n=100000, save_path=str(tmp_path / "simple"))
        Pslice = qs.cut_length(S, length=1e-7)
        assert len(P) == 100000
        assert P.charge == pytest.approx(S.charge)
        assert np.mean(P.pz) == pytest.approx(np.mean(S.pz), rel=1e-5)
        for key in ('x', 'px', 'y', 'py', 't'):
            assert np.std(getattr(P, key)) == pytest.approx(np.std(getattr(S, key)), rel=0.02), key
        assert np.std(P.pz) == pytest.approx(np.std(Pslice.pz), rel=0.02)  # slice energy spread
        # centered, and every correlation removed
        assert abs(np.mean(P.x)) < 3*np.std(S.x)/np.sqrt(len(P)) + 1e-9
        assert abs(np.mean(P.y)) < 3*np.std(S.y)/np.sqrt(len(P)) + 1e-9
        R = np.corrcoef(coords(P))
        assert np.allclose(R, np.eye(6), atol=0.02)
        assert (tmp_path / "simple.h5").exists()

    def test_correlated_matches_covariance(self, source_beam_file):
        base, S = source_beam_file
        np.random.seed(7)
        P = qs.make_simple_bunch(base, n=200000, save_path=base + "_corr", correlations=True)
        assert np.allclose(np.corrcoef(coords(P)), np.corrcoef(coords(S)), atol=0.01)
        assert np.allclose(np.std(coords(P), axis=1), np.std(coords(S), axis=1), rtol=0.02)
        assert P['norm_emit_x'] == pytest.approx(S['norm_emit_x'], rel=0.03)

    def test_default_n_and_save_name(self, source_beam_file):
        base, S = source_beam_file
        P = qs.make_simple_bunch(base)
        assert len(P) == len(S)
        assert os.path.exists(base + "_simple.h5")

    def test_flatter(self, source_beam_file):
        base, S = source_beam_file
        np.random.seed(8)
        P = qs.make_simple_bunch_flatter(base, n=100000, save_path=base + "_flat")
        assert np.std(P.t) == pytest.approx(np.std(S.t), rel=0.02)
        assert kurtosis(P.t) == pytest.approx(-0.81, abs=0.05)


class TestMakeSimpleBunchStandalone:
    def test_moments_with_zeros(self, tmp_path):
        np.random.seed(9)
        moments = [1e-3, 0.5e-3, 1e-3, 0, 1e-3, 0]
        P = qs.make_simple_bunch_standalone(N=1e5, meanPzMeV=125, moments=moments, means=[2e-3, 1e-4, 0, 0, 0, 0],
                                            charge=1e-9, save_path=str(tmp_path / "sa"))
        got = [np.std(P.x), np.std(P.xp), np.std(P.y), np.std(P.yp), C*np.std(P.t), np.std(P.pz)]
        assert np.allclose(got, moments, rtol=0.01, atol=0)
        assert np.mean(P.x) == pytest.approx(2e-3, abs=1e-5)
        assert np.mean(P.xp) == pytest.approx(1e-4, abs=3e-6)  # means are in rad, not eV/c
        assert np.all(P.pz == 125e6)
        assert P.charge == pytest.approx(1e-9)
        assert (tmp_path / "sa.h5").exists()

    def test_no_file_without_save_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        qs.make_simple_bunch_standalone(N=100)
        assert not any(tmp_path.iterdir())


class TestTheoryCoordinates:
    def test_order_and_units(self, tmp_path):
        n = 5
        P = ParticleGroup(data=dict(x=np.arange(n)*1e-6, px=np.full(n, 1e3), y=np.zeros(n), py=np.zeros(n), z=np.zeros(n),
                                    t=np.arange(n)*1e-15, pz=np.full(n, 101e6), status=np.ones(n, dtype=int),
                                    weight=np.full(n, 1e-12), species='electron'))
        P.write(str(tmp_path / "b.h5"))
        X = qs.make_simple_bunch_theory_from_bunch_sims(str(tmp_path / "b"), 100, means_shift=[1e-6, 0, 0, 0, 0, 0])
        assert X.shape == (n, 6)
        assert np.allclose(X[:, 0], P.x + 1e-6)
        assert np.allclose(X[:, 1], P.xp)
        assert np.allclose(X[:, 4], -3e8*P.t)
        assert np.allclose(X[:, 5], 0.01)  # (101 - 100)/100
