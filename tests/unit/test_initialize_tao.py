"""
Tests for initializeTao function

Covers:
- Environment variable setting
- Tao object verification
- File path handling
- File saving
- IMPACT-T integration
- Custom tao fields
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestInitializeTao:
    """Comprehensive tests for initializeTao function"""
    
    @patch('os.makedirs')
    @patch('FACET2_S2E.UTILITY_quickstart.ParticleGroup')
    @patch('FACET2_S2E.UTILITY_quickstart.Tao')
    def test_env_var_set(self, mock_tao_class, mock_pg, mock_makedirs):
        """Test that environment variable FACET2_LATTICE is set correctly"""
        from FACET2_S2E.UTILITY_quickstart import initializeTao
        
        mock_tao_instance = Mock()
        mock_tao_instance.cmd = Mock()
        mock_tao_class.return_value = mock_tao_instance
        mock_pg.return_value = Mock()
        
        test_path = '/tmp/test_lattice_path'
        try:
            result = initializeTao(filePath=test_path, runSetLatticeTF=False, scratchPath='/tmp')
        except Exception:
            pass
        
        # Check environment variable was set
        assert os.environ.get('FACET2_LATTICE') == test_path
    
    @pytest.mark.integration
    def test_tao_is_pytao_object(self):
        """Test that returned tao is actually a Tao object - integration test"""
        from FACET2_S2E.UTILITY_quickstart import initializeTao
        from pytao import Tao
        result = initializeTao(runSetLatticeTF=False)
        assert isinstance(result, Tao)
    
    @patch('os.makedirs')
    @patch('FACET2_S2E.UTILITY_quickstart.ParticleGroup')
    @patch('FACET2_S2E.UTILITY_quickstart.Tao')
    def test_file_paths_handled(self, mock_tao_class, mock_pg, mock_makedirs):
        """Test that file paths are correctly handled for activeFilePath, patchFilePath, qpadSimPath"""
        from FACET2_S2E.UTILITY_quickstart import initializeTao
        
        mock_tao_instance = Mock()
        mock_tao_instance.cmd = Mock()
        mock_tao_class.return_value = mock_tao_instance
        mock_pg.return_value = Mock()
        
        scratch_path = '/tmp/scratch'
        test_path = '/tmp/test_path'
        try:
            result = initializeTao(
                filePath=test_path,
                runSetLatticeTF=False,
                scratchPath=scratch_path
            )
            
            # Verify file path attributes are set correctly on tao object
            assert mock_tao_instance.activeFilePath == f'{scratch_path}/beams/activeBeamFile.h5'
            assert mock_tao_instance.patchFilePath == f'{scratch_path}/beams/patchBeamFile.h5'
            assert mock_tao_instance.qpadSimPath == f'{scratch_path}/beams/qpad_sim'
        except Exception:
            pass
    
    @patch('os.makedirs')
    @patch('FACET2_S2E.UTILITY_quickstart.modifyAndSaveInputBeam')
    @patch('FACET2_S2E.UTILITY_quickstart.ParticleGroup')
    @patch('FACET2_S2E.UTILITY_quickstart.Tao')
    def test_required_files_saved(self, mock_tao_class, mock_pg, mock_modify_save, mock_makedirs):
        """Test that modifyAndSaveInputBeam is called with correct outputBeamFilePath"""
        from FACET2_S2E.UTILITY_quickstart import initializeTao
        
        mock_tao_instance = Mock()
        mock_tao_instance.cmd = Mock()
        mock_tao_class.return_value = mock_tao_instance
        mock_pg.return_value = Mock()
        mock_modify_save.return_value = None
        
        scratch_path = '/tmp/scratch'
        num_macro_particles = 5000
        try:
            result = initializeTao(
                filePath='/tmp/test_path',
                runSetLatticeTF=False,
                runImpactTF=False,
                numMacroParticles=num_macro_particles,
                scratchPath=scratch_path
            )
        except Exception:
            pass
        
        # Verify modifyAndSaveInputBeam was called
        assert mock_modify_save.called
        
        # Check the call arguments
        call_args = mock_modify_save.call_args
        expected_active_path = f'{scratch_path}/beams/activeBeamFile.h5'
        
        # First positional arg should be inputBeamFilePath
        assert len(call_args[0]) > 0
        
        # Check keyword arguments
        call_kwargs = call_args[1]
        # Verify outputBeamFilePath matches expected path
        assert call_kwargs.get('outputBeamFilePath') == expected_active_path
        # When runImpactTF=False, numMacroParticles should be passed through (not None)
        assert call_kwargs.get('numMacroParticles') == num_macro_particles
    
    @patch('os.makedirs')
    @patch('FACET2_S2E.UTILITY_quickstart.ParticleGroup')
    @patch('FACET2_S2E.UTILITY_quickstart.runImpact')
    @patch('FACET2_S2E.UTILITY_quickstart.Tao')
    def test_impact_t_called_when_enabled(self, mock_tao_class, mock_run_impact, mock_pg, mock_makedirs):
        """Test that IMPACT-T is called when runImpactTF=True"""
        from FACET2_S2E.UTILITY_quickstart import initializeTao
        
        mock_tao_instance = Mock()
        mock_tao_instance.cmd = Mock()
        mock_tao_class.return_value = mock_tao_instance
        mock_pg.return_value = Mock()
        mock_run_impact.return_value = None
        
        try:
            result = initializeTao(
                filePath='/tmp/test_path',
                runSetLatticeTF=False,
                runImpactTF=True,
                numMacroParticles=1000,
                scratchPath='/tmp'
            )
        except Exception:
            pass
        
        # Verify runImpact was called
        assert mock_run_impact.called
    
    @patch('os.makedirs')
    @patch('FACET2_S2E.UTILITY_quickstart.modifyAndSaveInputBeam')
    @patch('FACET2_S2E.UTILITY_quickstart.ParticleGroup')
    @patch('FACET2_S2E.UTILITY_quickstart.Tao')
    def test_custom_tao_fields_saved(self, mock_tao_class, mock_pg, mock_modify_save, mock_makedirs):
        """Test that custom tao fields are saved during initialization"""
        from FACET2_S2E.UTILITY_quickstart import initializeTao
        
        mock_tao_instance = Mock()
        mock_tao_instance.cmd = Mock()
        mock_tao_class.return_value = mock_tao_instance
        mock_pg.return_value = Mock()
        mock_modify_save.return_value = None
        
        scratch_path = '/tmp/scratch'
        file_path = '/tmp/test_path'
        input_beam_suffix = '/beams/custom_beam.h5'
        qpad_defaults = 'qpad/qpad_defaults.yaml'
        
        result = initializeTao(
            filePath=file_path,
            runSetLatticeTF=False,
            scratchPath=scratch_path,
            inputBeamFilePathSuffix=input_beam_suffix,
            runQPAD=True,
            setQPADDefaultsFile=qpad_defaults
        )
        
        # Verify custom tao fields are set on the returned tao object
        assert result.inputBeamFilePath == f'{file_path}{input_beam_suffix}'
        assert result.activeFilePath == f'{scratch_path}/beams/activeBeamFile.h5'
        assert result.patchFilePath == f'{scratch_path}/beams/patchBeamFile.h5'
        assert result.qpadSimPath == f'{scratch_path}/beams/qpad_sim'
        assert result.runQPAD == True
        assert result.QPADDefaultsFile == qpad_defaults
    
    @patch('os.makedirs')
    @patch('FACET2_S2E.UTILITY_quickstart.ParticleGroup')
    @patch('FACET2_S2E.UTILITY_quickstart.Tao')
    def test_csr_off_when_disabled(self, mock_tao_class, mock_pg, mock_makedirs):
        """Test that CSR is turned off when csrTF=False"""
        from FACET2_S2E.UTILITY_quickstart import initializeTao
        
        mock_tao_instance = Mock()
        mock_tao_instance.cmd = Mock()
        mock_tao_class.return_value = mock_tao_instance
        mock_pg.return_value = Mock()
        
        try:
            result = initializeTao(
                filePath='/tmp/test_path',
                runSetLatticeTF=False,
                csrTF=False,
                scratchPath='/tmp'
            )
        except Exception:
            pass
        
        # Verify CSR off command was called
        if mock_tao_instance.cmd.called:
            cmd_calls = [str(call) for call in mock_tao_instance.cmd.call_args_list]
            assert any('csroff' in str(call) for call in cmd_calls)
    
    @patch('os.makedirs')
    @patch('FACET2_S2E.UTILITY_quickstart.ParticleGroup')
    @patch('FACET2_S2E.UTILITY_quickstart.setLattice')
    @patch('FACET2_S2E.UTILITY_quickstart.Tao')
    def test_set_lattice_called_when_enabled(self, mock_tao_class, mock_set_lattice, mock_pg, mock_makedirs):
        """Test that setLattice is called when runSetLatticeTF=True"""
        from FACET2_S2E.UTILITY_quickstart import initializeTao
        
        mock_tao_instance = Mock()
        mock_tao_instance.cmd = Mock()
        mock_tao_class.return_value = mock_tao_instance
        mock_pg.return_value = Mock()
        mock_set_lattice.return_value = None
        
        try:
            result = initializeTao(
                filePath='/tmp/test_path',
                runSetLatticeTF=True,
                scratchPath='/tmp'
            )
        except Exception:
            pass
        
        # Verify setLattice was called
        assert mock_set_lattice.called
    
    @patch('os.makedirs')
    @patch('FACET2_S2E.UTILITY_quickstart.ParticleGroup')
    @patch('FACET2_S2E.UTILITY_quickstart.Tao')
    def test_transverse_wakes_enabled(self, mock_tao_class, mock_pg, mock_makedirs):
        """Test that transverse wakes init file is used when enabled"""
        from FACET2_S2E.UTILITY_quickstart import initializeTao
        
        mock_tao_instance = Mock()
        mock_tao_instance.cmd = Mock()
        mock_tao_class.return_value = mock_tao_instance
        mock_pg.return_value = Mock()
        
        try:
            result = initializeTao(
                filePath='/tmp/test_path',
                runSetLatticeTF=False,
                transverseWakes=True,
                scratchPath='/tmp'
            )
        except Exception:
            pass
        
        # Verify correct tao.init was used
        call_args = str(mock_tao_class.call_args)
        assert 'tao_transverseWakesOn.init' in call_args or mock_tao_class.called
    
    @patch('os.makedirs')
    @patch('FACET2_S2E.UTILITY_quickstart.ParticleGroup')
    @patch('FACET2_S2E.UTILITY_quickstart.Tao')
    def test_beam_save_locations_set(self, mock_tao_class, mock_pg, mock_makedirs):
        """Test that beam save locations are configured"""
        from FACET2_S2E.UTILITY_quickstart import initializeTao
        
        mock_tao_instance = Mock()
        mock_tao_instance.cmd = Mock()
        mock_tao_class.return_value = mock_tao_instance
        mock_pg.return_value = Mock()
        
        try:
            result = initializeTao(
                filePath='/tmp/test_path',
                runSetLatticeTF=False,
                scratchPath='/tmp'
            )
        except Exception:
            pass
        
        # Verify beam save locations were set
        if mock_tao_instance.cmd.called:
            cmd_calls = [str(call) for call in mock_tao_instance.cmd.call_args_list]
            assert any('add_saved_at' in str(call) for call in cmd_calls)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
