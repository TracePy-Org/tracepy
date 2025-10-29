"""Performance tests for TracePy optimizations."""
import time
import numpy as np
import pytest
from tracepy import geometry
from tracepy.index import glass_index
from tracepy.raygroup import ray_plane


class TestGlassIndexPerformance:
    """Test glass index loading and caching performance."""
    
    def test_glass_loading_is_cached(self):
        """Verify that glass data is cached and loading is fast."""
        # First load should populate cache
        start = time.time()
        glass_func1 = glass_index("F2 schott")
        first_load = time.time() - start
        
        # Subsequent loads should be much faster due to caching
        start = time.time()
        for _ in range(10):
            glass_func = glass_index("F2 schott")
        cached_loads = (time.time() - start) / 10
        
        # Cached loads should be at least 5x faster than first load
        # (being conservative with the ratio to avoid flaky tests)
        assert cached_loads < first_load / 5, \
            f"Cached load ({cached_loads:.4f}s) should be much faster than first load ({first_load:.4f}s)"
    
    def test_glass_function_calls_are_fast(self):
        """Verify that glass function calls are fast."""
        glass_func = glass_index("F2 schott")
        wavelengths = np.array([0.45, 0.55, 0.65])
        
        # Should be able to call glass function many times quickly
        start = time.time()
        for _ in range(1000):
            result = glass_func(wavelengths)
        elapsed = time.time() - start
        
        # 1000 calls should take less than 100ms
        assert elapsed < 0.1, f"1000 glass function calls took {elapsed:.4f}s, should be < 0.1s"


class TestGeometryPerformance:
    """Test geometry calculation performance."""
    
    def test_conics_calculation_is_fast(self):
        """Verify that conics calculations are fast."""
        surface = {
            'P': 0.,
            'D': [0., 0., 0.],
            'action': 'refraction',
            'Diam': 25.4,
            'c': 1./50.,
            'kappa': 0.,
            'N': 1.5
        }
        geo = geometry(surface)
        point = np.array([1.0, 1.0, 0.0])
        
        # Should be able to call conics many times quickly
        start = time.time()
        for _ in range(1000):
            result = geo.conics(point)
        elapsed = time.time() - start
        
        # 1000 calls should take less than 50ms
        assert elapsed < 0.05, f"1000 conics calls took {elapsed:.4f}s, should be < 0.05s"
    
    def test_conics_plot_no_duplicate_calculation(self):
        """Verify that conics_plot doesn't duplicate rho calculation."""
        surface = {
            'P': 0.,
            'D': [0., 0., 0.],
            'action': 'refraction',
            'Diam': 25.4,
            'c': 1./50.,
            'kappa': 0.,
            'N': 1.5
        }
        geo = geometry(surface)
        
        # Create test points within aperture
        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        points = np.column_stack([x, y, np.zeros(100)])
        
        # Should be able to call conics_plot many times quickly
        start = time.time()
        for _ in range(100):
            result = geo.conics_plot(points)
        elapsed = time.time() - start
        
        # 100 calls with 100 points each should take less than 20ms
        assert elapsed < 0.02, f"100 conics_plot calls took {elapsed:.4f}s, should be < 0.02s"


class TestRayTracingPerformance:
    """Test overall ray tracing performance."""
    
    def test_ray_tracing_throughput(self):
        """Verify that ray tracing achieves good throughput."""
        lens1 = {
            'P': 0.,
            'D': [0., 0., 0.],
            'action': 'refraction',
            'Diam': 25.4,
            'c': 1./50.,
            'kappa': 0.,
            'N': 1.5
        }
        
        lens2 = {
            'P': 10.,
            'D': [0., 0., 0.],
            'action': 'refraction',
            'Diam': 25.4,
            'c': -1./50.,
            'kappa': 0.,
            'N': 1.0
        }
        
        stop = {
            'P': 100.,
            'D': [0., 0., 0.],
            'action': 'stop',
            'Diam': 25.4,
        }
        
        geo_params = [lens1, lens2, stop]
        
        # Trace 1000 rays through the system
        start = time.time()
        raygroup = ray_plane(geo_params, 0., 10., [0., 0., 1.], nrays=1000)
        elapsed = time.time() - start
        
        # 1000 rays through 3 surfaces should take less than 50ms
        assert elapsed < 0.05, f"Tracing 1000 rays took {elapsed:.4f}s, should be < 0.05s"
        
        # Verify rays were actually traced
        assert len(raygroup.P_hist) == 4, "Should have 4 positions in history (initial + 3 surfaces)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
