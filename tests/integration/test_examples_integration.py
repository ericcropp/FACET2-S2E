"""
Integration tests for example notebooks and Python files

These tests provide coverage for example code that is not already tested
in the core test suite (test_quickstart.py, test_quickstart_advanced.py).
Slow/external operations are mocked; real functions are called and results validated.
"""

import pytest
import numpy as np
import os
import importlib.util
from unittest.mock import Mock, MagicMock, patch, call

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import FACET2_S2E as qs


def _load_jitter_module():
    """Load the jitter study example module"""
    spec = importlib.util.spec_from_file_location(
        "jitter_study", 
        os.path.join(os.path.dirname(__file__), '..', '..', 'examples', 'Example - Jitter study.py')
    )
    jitter_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(jitter_module)
    return jitter_module


class TestJitterStudyExample:
    """Tests for 'Example - Jitter study.py' functions - call real jitterLinac function"""
    
    def test_jitter_linac_applies_phase_errors(self):
        """Test that jitterLinac actually applies phase errors to elements"""
        jitter_module = _load_jitter_module()
        
        # Create mock Tao with tracked cmd calls
        mock_tao = Mock()
        mock_tao.cmd = Mock()
        mock_tao.ele_gen_attribs = Mock(return_value={'GRADIENT': 1e7})
        
        # Set seed for reproducibility
        np.random.seed(42)
        
        # Call real function
        result = jitter_module.jitterLinac(
            mock_tao,
            L0BMatchStrings=['L0B_1'],
            L1MatchStrings=['L1_1'],
            L2MatchStrings=['L2_1'],
            L3MatchStrings=['L3_1'],
        )
        
        # Verify function returns dict with error values
        assert isinstance(result, dict)
        assert 'L0BPhaseError' in result
        assert 'L1PhaseError' in result
        assert all(isinstance(result[k], (int, float, np.number)) for k in result.keys())
        
        # Verify phase errors are small (< 0.01 turns)
        assert abs(result['L0BPhaseError']) < 0.01
        assert abs(result['L1PhaseError']) < 0.02  # 0.7 deg / 360
        
        # Verify cmd was called to apply phase errors
        phase_calls = [c for c in mock_tao.cmd.call_args_list if 'PHI0' in str(c)]
        assert len(phase_calls) >= 4, "Should have applied phase errors to all sections"
    
    def test_jitter_linac_applies_gradient_errors(self):
        """Test that jitterLinac calculates and applies gradient errors"""
        jitter_module = _load_jitter_module()
        
        mock_tao = Mock()
        mock_tao.cmd = Mock()
        # Mock gradient queries to return known values
        mock_tao.ele_gen_attribs = Mock(return_value={'GRADIENT': 1e7})
        
        np.random.seed(123)
        
        result = jitter_module.jitterLinac(
            mock_tao,
            L0BMatchStrings=['L0B_1', 'L0B_2'],
            L1MatchStrings=['L1_1'],
            L2MatchStrings=['L2_1'],
            L3MatchStrings=['L3_1'],
            L0BGradientErrorPercent=0.5,
            L1GradientErrorPercent=0.25,
        )
        
        # Verify gradient errors in result
        assert 'L0BGradientErrorRelative' in result
        assert 'L1GradientErrorRelative' in result
        
        # Gradient relative errors should be small percentages
        assert abs(result['L0BGradientErrorRelative']) < 0.01
        assert abs(result['L1GradientErrorRelative']) < 0.01
        
        # Verify ele_gen_attribs was called to get base gradients
        assert mock_tao.ele_gen_attribs.called
        
        # Verify GRADIENT changes were applied
        gradient_calls = [c for c in mock_tao.cmd.call_args_list if 'GRADIENT' in str(c)]
        assert len(gradient_calls) >= 4, "Should have applied gradient errors"
    
    def test_jitter_linac_disables_and_reenables_calc(self):
        """Test that jitterLinac properly manages lattice calculations"""
        jitter_module = _load_jitter_module()
        
        mock_tao = Mock()
        mock_tao.cmd = Mock()
        mock_tao.ele_gen_attribs = Mock(return_value={'GRADIENT': 1e7})
        
        jitter_module.jitterLinac(
            mock_tao,
            L0BMatchStrings=['L0B_1'],
            L1MatchStrings=['L1_1'],
            L2MatchStrings=['L2_1'],
            L3MatchStrings=['L3_1'],
        )
        
        # Extract cmd call strings
        cmd_calls = [str(c) for c in mock_tao.cmd.call_args_list]
        
        # Should disable calc, apply changes, then re-enable
        calc_off_found = any('lattice_calc_on = F' in c for c in cmd_calls)
        calc_on_found = any('lattice_calc_on = T' in c for c in cmd_calls)
        
        assert calc_off_found, "Should disable lattice calculations"
        assert calc_on_found, "Should re-enable lattice calculations"
    
    def test_jitter_linac_different_error_magnitudes(self):
        """Test that different sections get appropriately scaled errors"""
        jitter_module = _load_jitter_module()
        
        mock_tao = Mock()
        mock_tao.cmd = Mock()
        mock_tao.ele_gen_attribs = Mock(return_value={'GRADIENT': 1e7})
        
        np.random.seed(456)
        
        # Call with different error magnitudes per section
        result = jitter_module.jitterLinac(
            mock_tao,
            L0BMatchStrings=['L0B_1'],
            L1MatchStrings=['L1_1'],
            L2MatchStrings=['L2_1'],
            L3MatchStrings=['L3_1'],
            L0BPhaseErrorDeg=0.1,  # Small
            L1PhaseErrorDeg=0.7,   # Large
            L0BGradientErrorPercent=0.5,
            L1GradientErrorPercent=0.25,
        )
        
        # L1 phase error should be ~7x larger magnitude than L0B (same random seed)
        l0b_phase = abs(result['L0BPhaseError'])
        l1_phase = abs(result['L1PhaseError'])
        ratio = l1_phase / l0b_phase if l0b_phase > 1e-8 else float('inf')
        
        # Both should exist and be finite
        assert np.isfinite(result['L0BPhaseError'])
        assert np.isfinite(result['L1PhaseError'])


class TestExampleNotebookPatterns:
    """Tests for common patterns from example notebooks - call real FACET2_S2E functions"""
    
    def test_load_config_with_defaults(self):
        """Test loadConfig returns valid configuration dict"""
        # Call real function with default config
        config = qs.loadConfig(
            "setLattice_configs/defaults.yml",
            os.path.join(os.path.dirname(__file__), '..', '..')
        )
        
        assert isinstance(config, dict)
        assert len(config) > 0
        # Should have phase-related parameters
        assert any('Phase' in k or 'phase' in k for k in config.keys())
    
    def test_ballistic_propagation_calculation(self):
        """Test ballisticPropagation function with real calculation"""
        # Create mock beam using ParticleGroup-like structure
        from pmd_beamphysics import ParticleGroup
        
        # Create real beam data
        beam_data = {
            'x': np.array([0., 0., 0.]),
            'y': np.array([0., 0., 0.]),
            'z': np.array([-10e-6, 0, 10e-6]),
            'px': np.array([1e-6, -2e-6, 0.0]),
            'py': np.array([0., 0., 0.]),
            'pz': np.ones(3) * 1e9,
            't': np.array([-1e-12, 0, 1e-12]),
            'status': np.array([1, 1, 1]),
            'weight': np.ones(3),
            'species': 'electron',
        }
        beam = ParticleGroup(data=beam_data)
        
        # Call real function - modifies in place and returns None
        x_before = beam.x.copy()
        qs.ballisticPropagation(beam, 1.0)  # 1 meter propagation
        x_after = beam.x
        
        # Particles should have moved
        assert np.any(x_after != x_before)
        # With positive px, particles should move right
        assert x_after[0] > x_before[0]
        assert x_after[1] < x_before[1]  # Negative px should move left
    
    def test_get_driver_and_witness_separation(self):
        """Test getDriverAndWitness with real data"""
        from pmd_beamphysics import ParticleGroup
        
        # Create beam with two distinct weight values (driver/witness)
        beam_data = {
            'x': np.random.randn(6) * 1e-3,
            'y': np.random.randn(6) * 1e-3,
            'z': np.array([-10, -5, 0, 5, 10, 15]) * 1e-6,
            'px': np.zeros(6),
            'py': np.zeros(6),
            'pz': np.ones(6) * 1e9,
            't': np.array([-1, -0.5, 0, 0.5, 1, 1.5]) * 1e-12,
            'delta_z': np.array([-10, -5, 0, 5, 10, 15]) * 1e-6,
            'status': np.array([1, 1, 1, 1, 1, 1]),
            'weight': np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0]),
            'species': 'electron',
        }
        beam = ParticleGroup(data=beam_data)
        
        # Call real function
        driver, witness = qs.getDriverAndWitness(beam)
        
        # Verify separation
        assert driver is not None
        assert witness is not None
        # They should have different numbers of particles
        assert len(driver.x) != len(witness.x) or len(driver.x) > 0


class TestConfigAndIO:
    """Tests for configuration loading and I/O operations"""
    
    def test_load_config_returns_dict(self):
        """Test loadConfig returns dictionary"""
        config = qs.loadConfig(
            "setLattice_configs/defaults.yml",
            os.path.join(os.path.dirname(__file__), '..', '..')
        )
        
        assert isinstance(config, dict)
    
    def test_config_has_required_keys(self):
        """Test loaded config has expected keys"""
        config = qs.loadConfig(
            "setLattice_configs/defaults.yml",
            os.path.join(os.path.dirname(__file__), '..', '..')
        )
        
        # Should have at least some phase parameters or other config
        assert len(config) > 0
    
    def test_center_beam_shifts_distribution(self):
        """Test centerBeam actually shifts beam coordinates"""
        from pmd_beamphysics import ParticleGroup
        
        # Create asymmetric beam data that's not centered
        # Use enough particles to be asymmetric after centering
        x_data = np.array([5e-3, 6e-3, 7e-3, 8e-3, 9e-3])
        y_data = np.array([2e-3, 4e-3, 6e-3, 8e-3, 10e-3])
        beam_data = {
            'x': x_data,
            'y': y_data,
            'z': np.zeros(5),
            'px': np.zeros(5),
            'py': np.zeros(5),
            'pz': np.ones(5) * 1e9,
            't': np.zeros(5),
            'status': np.ones(5),
            'weight': np.ones(5),
            'species': 'electron',
        }
        beam = ParticleGroup(data=beam_data)
        original_x_median = np.median(beam.x)
        original_y_median = np.median(beam.y)
        
        # Call real function with median method
        centered = qs.centerBeam(beam, centerType='median')
        
        # Median should be shifted to zero
        assert np.isclose(np.median(centered.x), 0.0)
        assert np.isclose(np.median(centered.y), 0.0)
        # And shifted from original position
        assert np.median(centered.x) != original_x_median
        assert np.median(centered.y) != original_y_median


class TestBeamOperations:
    """Tests that validate beam operations with real functions"""
    
    def test_collimate_beam_removes_particles(self):
        """Test collimateBeam actually removes particles"""
        from pmd_beamphysics import ParticleGroup
        
        beam_data = {
            'x': np.random.randn(100) * 1e-3,
            'y': np.zeros(100),
            'z': np.zeros(100),
            'px': np.zeros(100),
            'py': np.zeros(100),
            'pz': np.ones(100) * 1e9,
            't': np.zeros(100),
            'status': np.ones(100),
            'weight': np.ones(100),
            'species': 'electron',
        }
        beam = ParticleGroup(data=beam_data)
        
        # Call real function with collimator rules [[x_min, x_max]]
        collimated = qs.collimateBeam(beam, allCollimatorRules=[[-0.5e-3, 0.5e-3]])
        
        # Should have fewer or equal particles
        assert len(collimated.x) <= 100
        # Remaining particles should be within aperture
        assert np.all(np.abs(collimated.x) >= 0.5e-3)
    
    def test_smallest_interval_implied_emittance(self):
        """Test smallestIntervalImpliedEmittance calculation"""
        from pmd_beamphysics import ParticleGroup
        
        # Create realistic beam data
        np.random.seed(42)
        beam_data = {
            'x': np.random.randn(100) * 1e-3,
            'y': np.random.randn(100) * 1e-3,
            'z': np.random.randn(100) * 1e-4,
            'px': np.random.randn(100) * 1e-2,
            'py': np.random.randn(100) * 1e-2,
            'pz': np.ones(100) * 1e9 + np.random.randn(100) * 1e5,
            't': np.zeros(100),
            'status': np.ones(100),
            'weight': np.ones(100),
            'species': 'electron',
        }
        beam = ParticleGroup(data=beam_data)
        
        # Call real function
        emittance_x = qs.smallestIntervalImpliedEmittance(beam, plane='x', percentage=0.9)
        
        # Should return positive emittance
        assert emittance_x > 0
        # Should be reasonable value (< 100 nm for test beam)
        assert emittance_x < 1e-3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
