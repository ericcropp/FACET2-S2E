"""
Pytest configuration and shared fixtures for FACET2_S2E tests
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock


@pytest.fixture
def mock_particle_group():
    """Create a mock ParticleGroup with realistic beam data"""
    P = Mock()
    n_particles = 1000
    
    # Position data
    P.x = np.random.normal(0, 1e-3, n_particles)
    P.y = np.random.normal(0, 1e-3, n_particles)
    P.z = np.random.normal(0, 100e-6, n_particles)
    
    # Momentum data
    P.px = np.random.normal(0, 1e6, n_particles)
    P.py = np.random.normal(0, 1e6, n_particles)
    P.pz = np.ones(n_particles) * 1e9
    
    # Particle properties
    P.t = np.zeros(n_particles)
    P.status = np.ones(n_particles, dtype=int)
    P.weight = np.ones(n_particles) * 1.6e-10
    P.species = 'electron'
    
    # Mock methods
    P.copy = Mock(return_value=P)
    P.__getitem__ = Mock(side_effect=lambda key: getattr(P, key))
    P.__setitem__ = Mock(side_effect=lambda key, val: setattr(P, key, val))
    
    # Statistical methods
    P.avg = Mock(return_value={
        'mean_x': 0,
        'mean_y': 0,
        'mean_z': 0,
        'mean_energy': 1e9
    })
    P.std = Mock(return_value={
        'sigma_x': 1e-3,
        'sigma_y': 1e-3,
        'sigma_z': 100e-6,
        'sigma_energy': 1e6
    })
    P.cov = Mock(return_value={
        'norm_emit_x': 1e-6,
        'norm_emit_y': 1e-6,
        'beta_x': 10.0,
        'beta_y': 10.0,
        'alpha_x': 0.0,
        'alpha_y': 0.0
    })
    
    return P


@pytest.fixture
def mock_two_bunch_particle_group():
    """Create a mock ParticleGroup with driver and witness bunches"""
    P = Mock()
    n_driver = 500
    n_witness = 500
    n_total = n_driver + n_witness
    
    # Position data with two bunches separated in z
    P.x = np.random.normal(0, 1e-3, n_total)
    P.y = np.random.normal(0, 1e-3, n_total)
    z_driver = np.random.normal(-150e-6, 20e-6, n_driver)
    z_witness = np.random.normal(150e-6, 20e-6, n_witness)
    P.z = np.concatenate([z_driver, z_witness])
    
    # Momentum data
    P.px = np.random.normal(0, 1e6, n_total)
    P.py = np.random.normal(0, 1e6, n_total)
    P.pz = np.ones(n_total) * 1e9
    
    # Particle properties
    P.t = np.zeros(n_total)
    P.status = np.ones(n_total, dtype=int)
    
    # Different weights for driver and witness
    weights = np.concatenate([
        np.ones(n_driver) * 1.6e-10,  # Driver
        np.ones(n_witness) * 1.6e-11  # Witness (10x lighter)
    ])
    P.weight = weights
    P.species = 'electron'
    
    # Mock methods
    P.copy = Mock(return_value=P)
    P.__getitem__ = Mock(side_effect=lambda key: getattr(P, key))
    
    return P


@pytest.fixture
def mock_tao():
    """Create a mock Tao object"""
    tao = Mock()
    
    # Mock basic attributes
    tao.filePathGlobal = '/tmp/test'
    tao.activeFilePath = '/tmp/test/beams/activeBeamFile.h5'
    tao.qpadSimPath = '/tmp/test/qpad'
    
    # Mock command interface
    tao.cmd = Mock(return_value=None)
    
    # Mock lattice data
    tao.lat_list = Mock(return_value=[
        'Q1', 'Q2', 'Q3', 'B1', 'B2', 'S1', 'S2', 'M1', 'M2'
    ])
    
    # Mock element data
    def mock_ele_gen_attribs(ele_name):
        return {
            'GRADIENT': 10.0,
            'B_FIELD': 0.5,
            'K1': 0.1,
            'PHI0': 0.0,
            'S': 100.0
        }
    tao.ele_gen_attribs = Mock(side_effect=mock_ele_gen_attribs)
    
    # Mock bunch data
    def mock_bunch_data(ele_string):
        return {
            'x': np.random.normal(0, 1e-3, 1000),
            'y': np.random.normal(0, 1e-3, 1000),
            'z': np.random.normal(0, 100e-6, 1000),
            'px': np.random.normal(0, 1e6, 1000),
            'py': np.random.normal(0, 1e6, 1000),
            'pz': np.ones(1000) * 1e9,
            't': np.zeros(1000),
            'charge': 1.6e-10,
            'species': 'electron',
            'status': np.ones(1000, dtype=int)
        }
    tao.bunch_data = Mock(side_effect=mock_bunch_data)
    
    # Mock matrix data
    def mock_matrix(ele1, ele2, order=1):
        # Return a 6x6 identity-like transfer matrix
        if order == 1:
            mat = np.eye(6)
            # Add some drift-like elements
            mat[0, 1] = 1.0  # x-px coupling
            mat[2, 3] = 1.0  # y-py coupling
            return mat
        return None
    tao.matrix = Mock(side_effect=mock_matrix)
    
    return tao


@pytest.fixture
def sample_yaml_config(tmp_path):
    """Create a sample YAML configuration file"""
    import yaml
    
    config_file = tmp_path / "test_config.yml"
    config_data = {
        'L0PhaseSet': -3.0,
        'L1PhaseSet': -25.4,
        'L2PhaseSet': -35.0,
        'L3PhaseSet': 0.0,
        'inputBeamFilePathSuffix': 'beams/example.h5',
        'quadSettings': {
            'Q1': 5.0,
            'Q2': -3.0,
            'Q3': 4.5
        }
    }
    
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)
    
    return config_file


@pytest.fixture
def sample_nested_yaml_config(tmp_path):
    """Create nested YAML configuration files"""
    import yaml
    
    # Base config
    base_config = tmp_path / "base_config.yml"
    base_data = {
        'baseKey1': 'baseValue1',
        'baseKey2': 42,
        'overrideKey': 'willBeOverridden'
    }
    with open(base_config, 'w') as f:
        yaml.dump(base_data, f)
    
    # Main config with include
    main_config = tmp_path / "main_config.yml"
    main_data = {
        'include': [str(base_config)],
        'mainKey': 'mainValue',
        'overrideKey': 'overriddenValue'
    }
    with open(main_config, 'w') as f:
        yaml.dump(main_data, f)
    
    return main_config, base_config


@pytest.fixture
def gaussian_beam_distribution():
    """Generate Gaussian beam distribution for testing"""
    n = 10000
    sigma_x = 1e-3
    sigma_y = 1e-3
    sigma_z = 100e-6
    
    data = {
        'x': np.random.normal(0, sigma_x, n),
        'y': np.random.normal(0, sigma_y, n),
        'z': np.random.normal(0, sigma_z, n),
        'px': np.random.normal(0, 1e6, n),
        'py': np.random.normal(0, 1e6, n),
        'pz': np.ones(n) * 1e9,
        't': np.zeros(n),
        'status': np.ones(n, dtype=int),
        'weight': np.ones(n) * 1.6e-10,
        'species': 'electron'
    }
    
    return data


@pytest.fixture
def mock_matplotlib():
    """Mock matplotlib to avoid display issues in tests"""
    import matplotlib
    matplotlib.use('Agg')
    return matplotlib


# Skip markers for tests requiring specific environments
def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line(
        "markers", "requires_tao: mark test as requiring Tao installation"
    )
    config.addinivalue_line(
        "markers", "requires_lattice: mark test as requiring FACET2 lattice files"
    )
    config.addinivalue_line(
        "markers", "requires_impact: mark test as requiring IMPACT-T"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
