"""
Tests for the spectrum / microbunching-gain functions in beam/microbunching.py.
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from pmd_beamphysics import ParticleGroup

import FACET2_S2E as qs
from FACET2_S2E.beam.microbunching import get_spec_band


C = 2.99792458e8
WAVELENGTH = 10e-6


def beam_from_z(z, charge=1e-9):
    n = len(z)
    return ParticleGroup(data=dict(x=np.zeros(n), px=np.zeros(n), y=np.zeros(n), py=np.zeros(n), z=np.zeros(n),
                                   t=z/C, pz=np.full(n, 1e9), status=np.ones(n, dtype=int),
                                   weight=np.full(n, charge/n), species='electron'))


def modulated_z(n, amplitude, wavelength=WAVELENGTH, length=300e-6, seed=0):
    """Flat-top bunch with density 1 + amplitude*cos(2 pi z / wavelength), by rejection sampling."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(0, length, 3*n)
    keep = rng.uniform(0, 1 + amplitude, 3*n) < 1 + amplitude*np.cos(2*np.pi*z/wavelength)
    return z[keep][:n]


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


class TestBinning:
    def test_bins_per_period(self):
        z = np.linspace(0, 100e-6, 10)
        assert qs.get_nbins_for_wavelength(z, 10e-6) == 160          # 100 um / (10 um / 16)
        assert qs.get_nbins_for_wavelength(z, 10e-6, bins_per_period=4) == 40

    def test_minimum_and_cap(self, capsys):
        z = np.linspace(0, 1e-6, 10)
        assert qs.get_nbins_for_wavelength(z, 1.0) == 2
        assert qs.get_nbins_for_wavelength(z, 1e-12, max_nbins=1000) == 1000
        assert "capping" in capsys.readouterr().out


class TestSpectrum:
    def test_peak_at_modulation_wavelength(self):
        P = beam_from_z(modulated_z(200000, 0.3))
        binwidth, (freq, fourier), sigz = qs.get_spectrum(P, min_wavelength=WAVELENGTH/2)
        assert sigz == pytest.approx(np.std(C*P.t), rel=1e-6)
        assert np.isrealobj(freq) and np.iscomplexobj(fourier)
        power = np.abs(fourier)**2
        k = np.argmax(power[5:len(power)//2]) + 5             # skip the DC / bunch-envelope peak
        assert binwidth/freq[k] == pytest.approx(WAVELENGTH, rel=0.02)

    def test_nbins_overrides_min_wavelength(self):
        P = beam_from_z(modulated_z(1000, 0.3))
        assert len(qs.get_spectrum(P, nbins=64, min_wavelength=1e-9)[1][0]) == 64

    def test_unresolved_band_raises(self):
        P = beam_from_z(modulated_z(1000, 0.3))
        spec = qs.get_spectrum(P, nbins=8)
        with pytest.raises(ValueError, match="Empty spectral band"):
            get_spec_band(spec, 2e-6, 1e-6)

    def test_analyze_spec_band_center(self):
        P = beam_from_z(modulated_z(200000, 0.3))
        spec = qs.get_spectrum(P, min_wavelength=5e-6)
        meanf, sigmaf, power = qs.analyze_spec(spec, 1.3*WAVELENGTH, 0.8*WAVELENGTH)
        assert spec[0]/meanf == pytest.approx(WAVELENGTH, rel=0.05)
        assert sigmaf > 0 and power > 0


class TestMicrobunchingGain:
    def test_gain(self, tmp_path):
        P0 = beam_from_z(modulated_z(200000, 0.02, seed=1))
        P1 = beam_from_z(modulated_z(200000, 0.2, seed=2))
        lmax, lmin = 1.3*WAVELENGTH, 0.8*WAVELENGTH
        assert qs.get_microbunching_gain_from_beams(P0, P0, lmax, lmin, lmax, lmin) == pytest.approx(1.0)
        gain = qs.get_microbunching_gain_from_beams(P0, P1, lmax, lmin, lmax, lmin, file=str(tmp_path / "g.txt"))
        assert 30 < gain < 300      # bunching factor x10 -> power x100 (plus shot noise)
        saved = np.loadtxt(str(tmp_path / "g.txt"))
        assert saved.shape == (2, 3) and saved[0, 2]/saved[1, 2] == pytest.approx(gain)

    def test_gain_from_files(self, tmp_path):
        P0 = beam_from_z(modulated_z(50000, 0.02, seed=1))
        P1 = beam_from_z(modulated_z(50000, 0.2, seed=2))
        P0.write(str(tmp_path / "a.h5"))
        P1.write(str(tmp_path / "b.h5"))
        args = (1.3*WAVELENGTH, 0.8*WAVELENGTH, 1.3*WAVELENGTH, 0.8*WAVELENGTH)
        assert qs.get_microbunching_gain(str(tmp_path / "a.h5"), str(tmp_path / "b.h5"), *args) == \
            pytest.approx(qs.get_microbunching_gain_from_beams(P0, P1, *args))


class TestMakeModulatedBunch:
    def test_modulation_and_charge(self, tmp_path):
        np.random.seed(3)
        P = beam_from_z(np.random.uniform(0, 300e-6, 20000), charge=2e-9)
        Q = qs.make_modulated_bunch(P, wavelength=WAVELENGTH, mod_amplitude=0.5, save_file=str(tmp_path / "m"))
        assert Q.charge == pytest.approx(2e-9)
        assert len(Q) == pytest.approx(0.75*len(P), rel=0.03)    # kept fraction 1 - amplitude/2
        assert (tmp_path / "m.h5").exists()
        # density 1 - (amplitude/2)(1 + sin kz) = 0.75 - 0.25 sin kz -> bunching factor 0.125/0.75 = 1/6
        b = np.abs(np.mean(np.exp(2j*np.pi*C*Q.t/WAVELENGTH)))
        assert b == pytest.approx(1/6, abs=0.02)


class TestPlots:
    def test_hist(self, monkeypatch):
        monkeypatch.setattr(plt, "show", lambda: None)
        qs.hist(np.random.normal(size=1000), xlim=(-3, 3))
        ax = plt.gcf().axes[0]
        assert ax.get_xlim() == (-3, 3)
        with pytest.raises(ValueError):
            qs.hist(np.zeros(10), xlim=(0,))

    def test_print_spec(self):
        P = beam_from_z(modulated_z(20000, 0.3))
        qs.print_spec(P, 1.3*WAVELENGTH, 0.8*WAVELENGTH)
        assert len(plt.gcf().axes) == 1
