"""
Tests for the beam-editing functions in beam/manipulation.py:
edit_bunch_parameters(_from_PG), cut_length, modifyInputBeamSimple.
"""

import inspect
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from pmd_beamphysics import ParticleGroup

import FACET2_S2E as qs
from FACET2_S2E.beam.manipulation import edit_bunch_parameters_from_PG, edit_bunch_parameters


C = 3e8


def r(a, b):
    return np.corrcoef(a, b)[0, 1]


def make_beam(n=20000, seed=0, rxxp=0.8, rzpz=0.7, mean_t=0.0, mean_x=0.0):
    """Beam with known correlations r(x, xp) and r(z, pz), z = -c*t."""
    rng = np.random.default_rng(seed)
    x = mean_x + rng.normal(0, 50e-6, n)
    xp = rxxp*(x - mean_x)/50e-6*20e-6 + np.sqrt(1 - rxxp**2)*rng.normal(0, 20e-6, n)
    y = rng.normal(0, 30e-6, n)
    yp = rng.normal(0, 10e-6, n)
    ct = rng.normal(0, 100e-6, n)
    pz = 125e6*(1 + 1e-3*(rzpz*(-ct)/100e-6 + np.sqrt(1 - rzpz**2)*rng.normal(size=n)))
    return ParticleGroup(data=dict(x=x, px=xp*125e6, y=y, py=yp*125e6, z=np.zeros(n), t=mean_t + ct/C, pz=pz,
                                   status=np.ones(n, dtype=int), weight=np.full(n, 1e-9/n), species='electron'))


class TestEditBunchParameters:
    def test_changing_sizes_keeps_correlations(self, tmp_path):
        P = make_beam()
        Q = edit_bunch_parameters_from_PG(P, moments=[100e-6, None, 60e-6, None, 200e-6, None],
                                          path_to_write=str(tmp_path / "e"))
        assert np.std(Q.x) == pytest.approx(100e-6, rel=1e-6)
        assert np.std(Q.y) == pytest.approx(60e-6, rel=1e-6)
        assert C*np.std(Q.t) == pytest.approx(200e-6, rel=1e-6)
        assert np.std(Q.xp) == pytest.approx(np.std(P.xp), rel=1e-3)   # unspecified size kept
        assert r(Q.x, Q.xp) == pytest.approx(r(P.x, P.xp), abs=1e-3)
        assert r(-Q.t, Q.pz) == pytest.approx(r(-P.t, P.pz), abs=1e-3)  # sign of the chirp kept
        assert (tmp_path / "e.h5").exists()

    def test_explicit_correlations(self, tmp_path):
        P = make_beam()
        Q = edit_bunch_parameters_from_PG(P, moments=[100e-6, 100e-6, 50e-6, 20e-6, 200e-6, 2e5],
                                          correlations=[0.9, -0.3, -0.5], path_to_write=str(tmp_path / "e"))
        assert r(Q.x, Q.xp) == pytest.approx(0.9, abs=1e-3)
        assert r(Q.y, Q.yp) == pytest.approx(-0.3, abs=1e-3)
        assert r(-Q.t, Q.pz) == pytest.approx(-0.5, abs=1e-3)
        assert np.std(Q.pz) == pytest.approx(2e5, rel=1e-3)

    def test_no_moments_leaves_distribution(self, tmp_path):
        P = make_beam()
        Q = edit_bunch_parameters_from_PG(P, means=[None]*5, path_to_write=str(tmp_path / "e"))
        for key in ('x', 'y', 't', 'pz'):
            assert np.allclose(getattr(Q, key), getattr(P, key), rtol=1e-9, atol=1e-30), key
        # px, py: the kept mean is <xp>*<pz>, which differs from <px> by cov(xp, pz)
        for key in ('px', 'py'):
            assert np.allclose(getattr(Q, key), getattr(P, key), rtol=0, atol=1e-4*np.std(getattr(P, key))), key

    def test_means(self, tmp_path):
        P = make_beam(mean_t=1e-12, mean_x=1e-3)
        Q = edit_bunch_parameters_from_PG(P, path_to_write=str(tmp_path / "e"))
        assert np.mean(Q.x) == pytest.approx(0, abs=1e-12)       # default means are 0
        assert C*np.mean(Q.t) == pytest.approx(0, abs=1e-12)
        Q = edit_bunch_parameters_from_PG(P, means=[None, 1e-4, 2e-3, None, 5e-4], path_to_write=str(tmp_path / "e"))
        assert np.mean(Q.x) == pytest.approx(np.mean(P.x), rel=1e-9)  # None keeps
        assert np.mean(Q.xp) == pytest.approx(1e-4, rel=1e-3)
        assert np.mean(Q.y) == pytest.approx(2e-3, rel=1e-9)
        assert C*np.mean(Q.t) == pytest.approx(5e-4, rel=1e-9)   # 5th mean is c*<t>

    def test_energy_and_charge(self, tmp_path):
        P = make_beam()
        Q = edit_bunch_parameters_from_PG(P, pzMeV=100, charge=2e-9, path_to_write=str(tmp_path / "e"))
        assert np.mean(Q.pz) == pytest.approx(100e6, rel=1e-9)
        assert Q.charge == pytest.approx(2e-9)
        assert np.std(Q.pz)/np.mean(Q.pz) == pytest.approx(np.std(P.pz)/np.mean(P.pz), rel=1e-6)

    def test_twiss_match(self, tmp_path):
        P = make_beam()
        Q = edit_bunch_parameters_from_PG(P, betaX=5.0, alphaX=-1.0, emittanceX=2e-9, path_to_write=str(tmp_path / "e"))
        twiss = Q.twiss('x')
        assert twiss['beta_x'] == pytest.approx(5.0, rel=1e-3)
        assert twiss['alpha_x'] == pytest.approx(-1.0, rel=1e-3)
        assert twiss['emit_x'] == pytest.approx(2e-9, rel=1e-3)

    def test_zero_spread_plane_is_filled(self, tmp_path):
        P = make_beam()
        P.y = np.zeros(len(P))
        P.py = np.zeros(len(P))
        Q = edit_bunch_parameters_from_PG(P, moments=[None, None, 40e-6, 4e-6, None, None], path_to_write=str(tmp_path / "e"))
        assert np.std(Q.y) == pytest.approx(40e-6, rel=1e-6)
        assert np.std(Q.yp) == pytest.approx(4e-6, rel=1e-3)

    def test_defaults_and_arguments_are_not_modified(self, tmp_path):
        before = {k: v.default for k, v in inspect.signature(edit_bunch_parameters_from_PG).parameters.items()}
        means = [0, 0, 0, 0, None]
        correlations = [None, None, None]
        moments = [100e-6, None, None, None, 200e-6, None]
        edit_bunch_parameters_from_PG(make_beam(mean_t=1e-12), moments=moments, correlations=correlations, means=means,
                                      path_to_write=str(tmp_path / "e"))
        assert means == [0, 0, 0, 0, None]
        assert correlations == [None, None, None]
        assert moments == [100e-6, None, None, None, 200e-6, None]
        after = {k: v.default for k, v in inspect.signature(edit_bunch_parameters_from_PG).parameters.items()}
        assert after == before

    def test_second_call_does_not_reuse_first_beam(self, tmp_path):
        # means[4]=None keeps <t>; it must be the <t> of each beam, not of the first one
        Q1 = edit_bunch_parameters_from_PG(make_beam(mean_t=1e-12), means=[0, 0, 0, 0, None], path_to_write=str(tmp_path / "e"))
        Q2 = edit_bunch_parameters_from_PG(make_beam(mean_t=5e-12), means=[0, 0, 0, 0, None], path_to_write=str(tmp_path / "e"))
        assert np.mean(Q1.t) == pytest.approx(1e-12, rel=1e-3)
        assert np.mean(Q2.t) == pytest.approx(5e-12, rel=1e-3)

    def test_file_wrapper(self, tmp_path):
        P = make_beam()
        P.write(str(tmp_path / "in.h5"))
        Q = edit_bunch_parameters(str(tmp_path / "in"), moments=[100e-6, None, None, None, None, None], path_to_write=str(tmp_path / "out"))
        R = ParticleGroup(str(tmp_path / "out.h5"))
        assert np.std(Q.x) == pytest.approx(100e-6, rel=1e-6)
        assert np.allclose(R.x, Q.x)


class TestCutLength:
    def test_half_width_window(self):
        P = make_beam(n=5000)
        L = 50e-6
        Q = qs.cut_length(P, L)
        Pd = P.copy()
        Pd.drift_to_z()
        expected = np.sum(np.abs(Pd.t - np.mean(Pd.t)) < L/C)
        assert len(Q) == expected
        assert np.all(np.abs(C*(Q.t - np.mean(Pd.t))) < L)
        assert 0 < len(Q) < len(P)

    def test_drift_back_to_t(self):
        P = make_beam(n=5000)
        Q = qs.cut_length(P, 50e-6, drift_to_z=False)
        assert np.ptp(Q.t) == pytest.approx(0, abs=1e-25)  # all particles at a common time

    def test_empty_window(self):
        with pytest.raises(ValueError, match="no particles within"):
            qs.cut_length(make_beam(n=100), 0)


class TestModifyInputBeamSimple:
    def test_subsample_and_center(self, tmp_path):
        P = make_beam(n=10000, mean_t=3e-12)
        P.write(str(tmp_path / "b.h5"))
        Q = qs.modifyInputBeamSimple(str(tmp_path / "b.h5"), numMacroParticles=1000)
        assert len(Q) == 1000
        assert Q.charge == pytest.approx(P.charge)
        assert np.all(Q.z == 0)
        assert np.mean(Q.t) == pytest.approx(0, abs=1e-20)

    def test_no_time_centering(self, tmp_path):
        P = make_beam(n=1000, mean_t=3e-12)
        P.write(str(tmp_path / "b.h5"))
        Q = qs.modifyInputBeamSimple(str(tmp_path / "b.h5"), timeCenterTF=False)
        assert len(Q) == 1000
        assert np.mean(Q.t) == pytest.approx(3e-12, rel=1e-6)
