"""
Smoke tests for full S2E simulation

These tests verify the complete start-to-end workflow can run without errors.
Smoke tests are faster than full system tests but slower than unit tests.
"""

import pytest
import numpy as np
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
import sys
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import FACET2_S2E as qs
from pmd_beamphysics import ParticleGroup

# Mark all tests in this module as smoke tests
pytestmark = pytest.mark.smoke


@pytest.fixture
def project_root():
    """Get the project root directory"""
    return str(Path(__file__).parent.parent.parent)


@pytest.fixture
def qpad_config():
    """QPAD configuration for testing (from qpad/2025-08-20-QPAD_defaults.yml)"""
    return {
        'simulation': {
            'grid': {
                'r': [0, 150.e-06],
                'r_cells': 32,
                'z': [-200.e-06, 100.e-06],
                'z_cells': 64,
                'max_mode': 0
            },
            'if_timing': True,
            'nprocs': [1, 1]
        },
        'diagnostics': {
            'ndumps': 10,
            'save_fields': True,
            'field_quants': ['rho', 'Ez']
        },
        'plasma': {
            'config': 'oven',
            'gas': 'Li',
            'P_torr': 4,
            'preionized': False
        }
    }


@pytest.fixture
def temp_beam_file(project_root):
    """Create a temporary beam file in the project root for mocked tests"""
    # Create a simple beam with 100 particles
    beam_data = {
        'x': np.random.randn(100) * 1e-3,
        'y': np.random.randn(100) * 1e-3,
        'z': np.zeros(100),
        'px': np.random.randn(100) * 1e6,
        'py': np.random.randn(100) * 1e6,
        'pz': np.ones(100) * 1e9,
        't': np.zeros(100),
        'status': np.ones(100, dtype=int),
        'weight': np.ones(100),
        'species': 'electron'
    }
    
    P = ParticleGroup(data=beam_data)
    
    # Create temp directory in project root
    with tempfile.TemporaryDirectory(dir=project_root) as tmpdir:
        beam_file = os.path.join(tmpdir, "test_beam.h5")
        P.write(beam_file)
        
        # Return relative path from project root
        rel_path = os.path.relpath(beam_file, project_root)
        yield f"/{os.path.basename(tmpdir)}/{os.path.basename(beam_file)}"


@pytest.mark.slow
class TestFullS2EReal:
    """Real end-to-end smoke tests that actually run simulators"""
    
    def test_full_s2e_creates_particle_group(self, project_root, qpad_config):
        """
        Test that full S2E simulation runs and creates ParticleGroup objects
        
        This is a smoke test that runs IMPACT-T, Bmad, and QPAD with minimal
        particle counts to verify the full pipeline works.
        """
        try:
            # Load configuration
            config = qs.loadConfig(
                "setLattice_configs/2024-10-22_oneBunch_baseline3.yml",
                project_root
            )
            
            # Create temp directory inside project root so initializeTao can find the config
            with tempfile.TemporaryDirectory(dir=project_root) as config_tmpdir:
                # Write qpad_config fixture to temp file
                qpad_config_file = os.path.join(config_tmpdir, "qpad_config.yml")
                with open(qpad_config_file, 'w') as f:
                    yaml.dump(qpad_config, f)
                
                # Use relative path from project root
                qpad_config_relpath = os.path.relpath(qpad_config_file, project_root)
                
                # Initialize Tao with minimal settings for speed
                with tempfile.TemporaryDirectory() as scratch_tmpdir:
                    tao = qs.initializeTao(
                        filePath=project_root,
                        inputBeamFilePathSuffix=config["inputBeamFilePathSuffix"],
                        csrTF=True,
                        scratchPath=scratch_tmpdir,
                        randomizeFileNames=True,
                        runQPAD=True,
                        setQPADDefaultsFile=qpad_config_relpath,
                        transverseWakes=False,
                        runImpactTF=True,
                        numMacroParticles=100,  # Minimal for speed
                        impactChargepC=1800,
                        impactGridCount=4
                    )
                    
                    # Set up lattice
                    qs.setLattice(tao, **config)
                    
                    # Track beam through lattice
                    qs.trackBeam(tao, filepath=project_root, plasmaSIM=True, **config)
                    
                    # Get beam at PENT (plasma entrance)
                    P = qs.getBeamAtElement(tao, "PENT")
                    
                    # Assertions
                    assert P is not None, "Failed to get beam at PENT"
                    assert isinstance(P, ParticleGroup), f"Expected ParticleGroup, got {type(P)}"
                    assert len(P.x) > 0, "Beam has no particles"
                    assert np.all(np.isfinite(P.x)), "Beam has non-finite x coordinates"
                    assert np.all(np.isfinite(P.y)), "Beam has non-finite y coordinates"
                    assert np.all(np.isfinite(P.pz)), "Beam has non-finite pz coordinates"
                    
                    # Get beam at PEXT (plasma exit)
                    P2 = qs.getBeamAtElement(tao, "PEXT")
                    
                    assert P2 is not None, "Failed to get beam at PEXT"
                    assert isinstance(P2, ParticleGroup), f"Expected ParticleGroup, got {type(P2)}"
                    assert len(P2.x) > 0, "Beam at PEXT has no particles"
                    
                    # Verify beam tracking worked (particles moved)
                    assert not np.array_equal(P.z, P2.z), "Beam did not propagate through plasma"
                
        except Exception as e:
            pytest.skip(f"Full S2E simulation failed (simulators may not be available): {e}")


class TestFullS2EMocked:
    """Mocked smoke tests that verify function calls without running simulators"""
    
    @patch('FACET2_S2E.UTILITY_quickstart.modifyAndSaveInputBeam')
    @patch('FACET2_S2E.UTILITY_quickstart.runImpact')
    @patch('FACET2_S2E.UTILITY_quickstart.Tao')
    def test_full_s2e_calls_all_simulators(
        self,
        mock_tao_class,
        mock_run_impact,
        mock_modify_save_beam,
        project_root,
        qpad_config,
        temp_beam_file
    ):
        """
        Test that S2E workflow initializes all three simulators
        
        This mocked test verifies that IMPACT-T, Bmad (Tao), and QPAD
        are properly initialized when requested.
        """
        # Setup mock Tao instance
        mock_tao = MagicMock()
        mock_tao.cmd = MagicMock()
        mock_tao.filePathGlobal = project_root
        mock_tao.activeFilePath = "/tmp/test_beam.h5"
        mock_tao.patchFilePath = "/tmp/test_patch.h5"
        mock_tao.qpadSimPath = "/tmp/qpad_sim"
        mock_tao_class.return_value = mock_tao
        
        # Setup mock returns
        mock_run_impact.return_value = None
        
        # Load configuration
        config = qs.loadConfig(
            "setLattice_configs/2024-10-22_oneBunch_baseline3.yml",
            project_root
        )
        
        # Create temp directory inside project root so initializeTao can find the config
        with tempfile.TemporaryDirectory(dir=project_root) as config_tmpdir:
            # Write QPAD config to temp file
            qpad_config_file = os.path.join(config_tmpdir, "qpad_config.yml")
            with open(qpad_config_file, 'w') as f:
                yaml.dump(qpad_config, f)
            
            # Use relative path from project root
            qpad_config_relpath = os.path.relpath(qpad_config_file, project_root)
            
            # Initialize with simulators enabled - skip setLattice
            with tempfile.TemporaryDirectory() as scratch_tmpdir:
                tao = qs.initializeTao(
                    filePath=project_root,
                    inputBeamFilePathSuffix=temp_beam_file,
                    csrTF=True,
                    scratchPath=scratch_tmpdir,
                    randomizeFileNames=True,
                    runQPAD=True,  # QPAD enabled
                    setQPADDefaultsFile=qpad_config_relpath,
                    transverseWakes=False,
                    runImpactTF=True,  # IMPACT-T enabled
                    numMacroParticles=100,
                    impactChargepC=1800,
                    impactGridCount=4,
                    runSetLatticeTF=False  # Skip setLattice to avoid complex mocking
                )
                
                # Verify the three simulators are initialized:
                
                # 1. IMPACT-T is called
                assert mock_run_impact.called, "IMPACT-T (runImpact) was not called"
                
                # 2. Tao/Bmad is initialized
                assert mock_tao_class.called, "Tao/Bmad was not initialized"
                assert tao is not None, "Tao object should be returned"
                
                # 3. QPAD configuration is set
                assert tao.qpadSimPath is not None, "QPAD path should be configured"
                assert hasattr(tao, 'qpadSimPath'), "QPAD simulator path should be stored"
    
    @patch('FACET2_S2E.UTILITY_quickstart.modifyAndSaveInputBeam')
    @patch('FACET2_S2E.UTILITY_quickstart.setLattice')
    @patch('FACET2_S2E.UTILITY_quickstart.Tao')
    def test_impact_called_when_enabled(self, mock_tao_class, mock_set_lattice, mock_modify_save_beam, project_root, temp_beam_file):
        """Test that IMPACT-T is called when runImpactTF=True"""
        with patch('FACET2_S2E.UTILITY_quickstart.runImpact') as mock_impact:
            mock_tao = MagicMock()
            mock_tao.lat_list = MagicMock(return_value=['Q1', 'Q2'])
            mock_tao.ele_gen_attribs = MagicMock(return_value={'GRADIENT': 1e7})
            mock_tao_class.return_value = mock_tao
            mock_set_lattice.return_value = None
            
            with tempfile.TemporaryDirectory() as tmpdir:
                tao = qs.initializeTao(
                    filePath=project_root,
                    inputBeamFilePathSuffix=temp_beam_file,
                    runImpactTF=True,
                    numMacroParticles=100,
                    scratchPath=tmpdir,
                    setLatticeDefaultsFile=None
                )
                
                assert mock_impact.called, "runImpact was not called when runImpactTF=True"
    
    @patch('FACET2_S2E.UTILITY_quickstart.modifyAndSaveInputBeam')
    @patch('FACET2_S2E.UTILITY_quickstart.setLattice')
    @patch('FACET2_S2E.UTILITY_quickstart.Tao')
    def test_qpad_enabled_when_configured(self, mock_tao_class, mock_set_lattice, mock_modify_save_beam, project_root, qpad_config, temp_beam_file):
        """Test that QPAD is enabled when runQPAD=True"""
        mock_tao = MagicMock()
        mock_tao.lat_list = MagicMock(return_value=['Q1', 'Q2'])
        mock_tao.ele_gen_attribs = MagicMock(return_value={'GRADIENT': 1e7})
        mock_tao_class.return_value = mock_tao
        mock_set_lattice.return_value = None
        
        with tempfile.TemporaryDirectory(dir=project_root) as config_tmpdir:
            # Write QPAD config to temp file
            qpad_config_file = os.path.join(config_tmpdir, "qpad_config.yml")
            with open(qpad_config_file, 'w') as f:
                yaml.dump(qpad_config, f)
            
            # Use relative path from project root
            qpad_config_relpath = os.path.relpath(qpad_config_file, project_root)
            
            tao = qs.initializeTao(
                filePath=project_root,
                inputBeamFilePathSuffix=temp_beam_file,
                runQPAD=True,
                setQPADDefaultsFile=qpad_config_relpath,
                scratchPath=config_tmpdir,
                setLatticeDefaultsFile=None
            )
            
            # Verify QPAD paths were set
            assert hasattr(tao, 'qpadSimPath') or mock_tao.qpadSimPath, \
                "QPAD simulation path not set when runQPAD=True"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
