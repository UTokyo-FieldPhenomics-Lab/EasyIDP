"""
Tests for geotools module (including subplot generation).
"""

import pytest
import numpy as np
import pyproj
import tempfile
from pathlib import Path

import easyidp as idp


@pytest.fixture
def rectangular_boundary():
    """Create a simple rectangular boundary ROI for testing."""
    roi = idp.ROI()
    # 10x20 meter rectangle
    roi["test_boundary"] = np.array([
        [0, 0],
        [20, 0],
        [20, 10],
        [0, 10],
        [0, 0],
    ], dtype=float)
    roi.crs = pyproj.CRS.from_epsg(32654)  # UTM 54N
    return roi


@pytest.fixture
def l_shaped_boundary():
    """Create an L-shaped boundary for testing non-rectangular cases."""
    roi = idp.ROI()
    # L-shape: 20x10 with 10x5 cut out from top-right
    roi["test_l_boundary"] = np.array([
        [0, 0],
        [20, 0],
        [20, 5],
        [10, 5],
        [10, 10],
        [0, 10],
        [0, 0],
    ], dtype=float)
    roi.crs = pyproj.CRS.from_epsg(32654)
    return roi


class TestGenerateSubplotsGridMode:
    """Tests for grid-based subplot generation."""

    def test_basic_grid_generation(self, rectangular_boundary):
        """Test basic 2x4 grid generation."""
        subplots = idp.geotools.generate_subplots(
            rectangular_boundary,
            row_num=2,
            col_num=4,
            x_interval=0,
            y_interval=0,
        )
        # Should generate 2*4=8 subplots
        assert len(subplots) == 8
        assert subplots.crs == rectangular_boundary.crs

    def test_grid_with_interval(self, rectangular_boundary):
        """Test grid generation with intervals."""
        subplots = idp.geotools.generate_subplots(
            rectangular_boundary,
            row_num=2,
            col_num=4,
            x_interval=0.5,
            y_interval=0.5,
        )
        assert len(subplots) == 8

        # Check naming convention
        assert "R1C1" in subplots.keys()
        assert "R2C4" in subplots.keys()

    def test_grid_naming_convention(self, rectangular_boundary):
        """Test that subplots are named R{row}C{col}."""
        subplots = idp.geotools.generate_subplots(
            rectangular_boundary,
            row_num=3,
            col_num=2,
        )
        expected_names = ["R1C1", "R1C2", "R2C1", "R2C2", "R3C1", "R3C2"]
        for name in expected_names:
            assert name in subplots.keys(), f"Missing subplot {name}"

    def test_subplot_metadata(self, rectangular_boundary):
        """Test that subplot metadata is stored correctly."""
        subplots = idp.geotools.generate_subplots(
            rectangular_boundary,
            row_num=2,
            col_num=2,
        )
        assert hasattr(subplots, '_subplot_meta')
        meta = subplots._subplot_meta["R1C1"]
        assert meta['row'] == 1
        assert meta['col'] == 1
        assert meta['status'] in ('inside', 'touch', 'outside')


class TestGenerateSubplotsSizeMode:
    """Tests for size-based subplot generation."""

    def test_basic_size_generation(self, rectangular_boundary):
        """Test subplot generation by size (5m x 5m)."""
        subplots = idp.geotools.generate_subplots(
            rectangular_boundary,
            width=5,
            height=5,
        )
        # 20m width / 5m = 4 cols, 10m height / 5m = 2 rows
        assert len(subplots) == 8

    def test_size_with_interval(self, rectangular_boundary):
        """Test size mode with intervals."""
        subplots = idp.geotools.generate_subplots(
            rectangular_boundary,
            width=4,
            height=4,
            x_interval=1,
            y_interval=1,
        )
        # 20m / (4+1) = 4 cols, 10m / (4+1) = 2 rows
        assert len(subplots) == 8

    def test_size_not_fitting_evenly(self, rectangular_boundary):
        """Test when dimensions don't divide evenly."""
        subplots = idp.geotools.generate_subplots(
            rectangular_boundary,
            width=7,
            height=7,
        )
        # 20m / 7m = 2 cols (floor), 10m / 7m = 1 row (floor)
        assert len(subplots) >= 1


class TestGenerateSubplotsKeepMode:
    """Tests for keep mode filtering with non-rectangular boundaries."""

    def test_keep_all(self, l_shaped_boundary):
        """Test keep='all' includes all subplots in MAR."""
        subplots = idp.geotools.generate_subplots(
            l_shaped_boundary,
            row_num=2,
            col_num=4,
            keep="all",
        )
        # All subplots in MAR should be included
        assert len(subplots) == 8

    def test_keep_touch(self, l_shaped_boundary):
        """Test keep='touch' excludes completely outside subplots."""
        # Use finer grid (4x4) to ensure some subplots are outside L-shape
        subplots = idp.geotools.generate_subplots(
            l_shaped_boundary,
            row_num=4,
            col_num=4,
            keep="touch",
        )
        # All remaining should be inside or touch (no outside)
        for name in subplots.keys():
            status = subplots._subplot_meta[name]['status']
            assert status in ('inside', 'touch'), f"{name} has status {status}"

    def test_keep_inside(self, l_shaped_boundary):
        """Test keep='inside' only keeps fully contained subplots."""
        subplots = idp.geotools.generate_subplots(
            l_shaped_boundary,
            row_num=2,
            col_num=4,
            keep="inside",
        )
        # Only fully inside subplots
        for name in subplots.keys():
            status = subplots._subplot_meta[name]['status']
            assert status == 'inside'


class TestGenerateSubplotsValidation:
    """Tests for input validation."""

    def test_empty_boundary_raises(self):
        """Test that empty boundary raises ValueError."""
        empty_roi = idp.ROI()
        empty_roi.crs = pyproj.CRS.from_epsg(32654)

        with pytest.raises(ValueError, match="empty"):
            idp.geotools.generate_subplots(empty_roi, row_num=2, col_num=2)

    def test_multiple_polygons_raises(self, rectangular_boundary):
        """Test that multiple polygons raise ValueError."""
        # Add second polygon
        rectangular_boundary["second"] = np.array([
            [100, 100],
            [110, 100],
            [110, 110],
            [100, 110],
            [100, 100],
        ])

        with pytest.raises(ValueError, match="exactly one"):
            idp.geotools.generate_subplots(rectangular_boundary, row_num=2, col_num=2)

    def test_grid_and_size_mutually_exclusive(self, rectangular_boundary):
        """Test that grid and size modes cannot be combined."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            idp.geotools.generate_subplots(
                rectangular_boundary,
                row_num=2,
                col_num=2,
                width=5,
                height=5,
            )

    def test_missing_parameters_raises(self, rectangular_boundary):
        """Test that missing required parameters raise ValueError."""
        with pytest.raises(ValueError, match="Must specify"):
            idp.geotools.generate_subplots(rectangular_boundary)

        with pytest.raises(ValueError, match="requires both row_num and col_num"):
            idp.geotools.generate_subplots(rectangular_boundary, row_num=2)

        with pytest.raises(ValueError, match="requires both width and height"):
            idp.geotools.generate_subplots(rectangular_boundary, width=5)

    def test_invalid_keep_mode_raises(self, rectangular_boundary):
        """Test that invalid keep mode raises ValueError."""
        with pytest.raises(ValueError, match="keep must be"):
            idp.geotools.generate_subplots(
                rectangular_boundary,
                row_num=2,
                col_num=2,
                keep="invalid",
            )

    def test_negative_values_raise(self, rectangular_boundary):
        """Test that negative/zero dimensions raise ValueError."""
        with pytest.raises(ValueError, match="must be >= 1"):
            idp.geotools.generate_subplots(rectangular_boundary, row_num=0, col_num=2)

        with pytest.raises(ValueError, match="must be > 0"):
            idp.geotools.generate_subplots(rectangular_boundary, width=-1, height=5)


class TestROISaveShp:
    """Tests for ROI.save_shp() method (which now uses idp.shp.write_shp)."""

    def test_save_and_reload(self, rectangular_boundary):
        """Test save and reload roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shp_path = Path(tmpdir) / "test_output.shp"
            
            # Save
            result_path = rectangular_boundary.save_shp(shp_path)
            
            assert result_path.exists()
            assert (result_path.with_suffix('.prj')).exists()
            assert (result_path.with_suffix('.dbf')).exists()
            assert (result_path.with_suffix('.shx')).exists()

            # Reload and verify
            reloaded = idp.ROI(str(result_path))
            assert len(reloaded) == 1

    def test_save_subplots_with_metadata(self, rectangular_boundary):
        """Test saving subplots preserves metadata."""
        subplots = idp.geotools.generate_subplots(
            rectangular_boundary,
            row_num=2,
            col_num=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            shp_path = Path(tmpdir) / "subplots.shp"
            subplots.save_shp(shp_path)

            assert shp_path.exists()

    def test_save_empty_raises(self):
        """Test that saving empty ROI raises ValueError."""
        empty_roi = idp.ROI()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            shp_path = Path(tmpdir) / "empty.shp"
            
            with pytest.raises(ValueError, match="empty"):
                empty_roi.save_shp(shp_path)

    def test_save_generic_method(self, rectangular_boundary):
        """Test the new ROI.save() method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shp_path = Path(tmpdir) / "generic.shp"
            
            # Save using .save()
            result_path = rectangular_boundary.save(shp_path)
            assert result_path.exists()
