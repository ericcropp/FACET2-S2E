"""
Tests for plotting: the merged plotMod (standalone / embedded / z_from_t), print_result*, make_a_plot, enable_plt_styling.
"""

import os
import sys
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from pmd_beamphysics import ParticleGroup

import FACET2_S2E as qs


@pytest.fixture
def beam():
    """Bmad-like beam at a fixed s: z = 0 for all particles, spread in t."""
    rng = np.random.default_rng(0)
    n = 5000
    t = rng.normal(0, 30e-15, n)
    return ParticleGroup(data=dict(x=rng.normal(0, 20e-6, n), px=rng.normal(0, 1e3, n), y=rng.normal(0, 10e-6, n),
                                   py=rng.normal(0, 1e3, n), z=np.zeros(n), t=t, pz=10e9*(1 + 1e-3*rng.normal(size=n)),
                                   status=np.ones(n, dtype=int), weight=np.full(n, 1.6e-9/n), species='electron'))


@pytest.fixture(autouse=True)
def clean_pyplot_state():
    was_interactive = plt.isinteractive()
    yield
    plt.close('all')
    plt.interactive(was_interactive)


class TestPlotModStandalone:
    def test_returns_figure_with_three_axes(self, beam):
        plt.figure()
        plt.figure()
        fig = qs.plotMod(beam, 'x', 'y', bins=50)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == 3
        assert plt.get_fignums() == [fig.number]         # previous figures closed
        assert 'x' in fig.axes[0].get_xlabel() and 'µm' in fig.axes[0].get_xlabel()
        assert 'y' in fig.axes[0].get_ylabel()

    def test_leaves_interactive_off(self, beam):
        # so that display(qs.plotMod(...)) shows the figure exactly once
        plt.ion()
        qs.plotMod(beam, 'x', 'y', bins=50)
        assert not plt.isinteractive()

    def test_limits(self, beam):
        fig = qs.plotMod(beam, 't', 'pz', bins=50, xlim=(-50e-15, 50e-15))
        assert fig.axes[0].get_xlim() == pytest.approx((-50, 50))   # in fs
        assert fig.axes[1].get_ylabel() == 'kA'                      # current for a time axis


class TestPlotModEmbedded:
    def test_draws_into_cell_and_restores_state(self, beam):
        plt.ion()
        other = plt.figure()
        fig = plt.figure()
        outer = GridSpec(1, 2)
        out = qs.plotMod(beam, 'x', 'xp', bins=50, fig=fig, outer=outer, i=1)
        assert out is fig
        assert len(fig.axes) == 3
        assert plt.isinteractive()                         # restored
        assert plt.fignum_exists(other.number)             # nothing else closed
        ax_joint = fig.axes[0]
        assert ax_joint.get_position().x0 > 0.5            # right-hand cell

    def test_background(self, beam):
        fig = plt.figure()
        qs.plotMod(beam, 'x', 'y', bins=50, fig=fig, outer=GridSpec(1, 1), i=0, background='white')
        assert fig.axes[0].get_facecolor()[:3] == (1.0, 1.0, 1.0)


class TestPlotModZFromT:
    @pytest.mark.parametrize("keys, axis", [(('z', 'pz'), 'x'), (('pz', 'z'), 'y')])
    def test_z_axis(self, beam, keys, axis):
        fig = qs.plotMod(beam, *keys, bins=50, z_from_t=True)
        ax_joint = fig.axes[0]
        label = ax_joint.get_xlabel() if axis == 'x' else ax_joint.get_ylabel()
        lim = ax_joint.get_xlim() if axis == 'x' else ax_joint.get_ylim()
        assert 'z' in label and 'µm' in label
        # z = -c t: about +-4 sigma of 9 um
        sz = 299792458*np.std(beam.t)*1e6
        assert 2*sz < max(abs(lim[0]), abs(lim[1])) < 8*sz
        marginal = fig.axes[1].get_ylabel() if axis == 'x' else fig.axes[2].get_xlabel()
        assert 'C/' in marginal and 'µm' in marginal

    def test_z_data_is_minus_ct(self, beam):
        fig = qs.plotMod(beam, 'z', 'x', bins=50, z_from_t=True, nice=False)
        hexbin = fig.axes[0].collections[0]
        offsets = hexbin.get_offsets()
        assert np.average(offsets[:, 0], weights=hexbin.get_array()) == pytest.approx(
            -299792458*np.mean(beam.t), abs=1e-6)


class TestPrintResult:
    def test_axes_and_moments(self, beam, capsys):
        qs.print_result(beam, couples=[['x', 'xp'], ['z', 'pz']], z_from_t=True)
        fig = plt.gcf()
        assert len(fig.axes) == 6
        printed = capsys.readouterr().out.strip().splitlines()[-1]
        values = [float(v) for v in printed.strip('[]').split(',')]
        assert len(values) == 7
        assert values[0] == pytest.approx(np.std(beam.x), rel=1e-6)
        assert values[6] == pytest.approx(np.std(beam.t), rel=1e-6)

    def test_from_file(self, beam, tmp_path, capsys):
        beam.write(str(tmp_path / "b.h5"))
        qs.print_result_from_file(str(tmp_path / "b.h5"), couples=[['x', 'y']], moments=False)
        assert len(plt.gcf().axes) == 3
        assert capsys.readouterr().out.strip() == ''

    def test_from_tao(self, beam):
        with patch('FACET2_S2E.plotting.bunch_summary.getBeamAtElement', return_value=beam) as get:
            qs.print_result_from_tao('tao', 'PENT', couples=[['x', 'y'], ['t', 'pz']], moments=False)
        get.assert_called_once_with('tao', 'PENT', tToZ=False)
        assert len(plt.gcf().axes) == 6


class TestStyle:
    def test_make_a_plot_single_and_twin_axes(self):
        x = np.linspace(0, 1, 20)
        qs.make_a_plot(x, np.sin(x), label="s", x_label="x", y_label="dz")    # 2-character label: one axis
        assert len(plt.gcf().axes) == 1
        assert plt.gcf().axes[0].get_ylabel() == "dz"
        plt.close('all')
        qs.make_a_plot([x, x], [np.sin(x), np.cos(x)], label=["s", "c"], x_label="x", y_label=["sin", "cos"])
        assert len(plt.gcf().axes) == 2

    def test_enable_plt_styling(self):
        with plt.rc_context():
            qs.enable_plt_styling()
            assert plt.rcParams["font.family"] == ["serif"]
            assert plt.rcParams["xtick.direction"] == "in"
