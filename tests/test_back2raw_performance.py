"""
Performance comparison tests for back2raw vs back2raw_batch.

This module tests the optimized batch version of backward projection
to ensure correctness and measure speedup.
"""
import time
import pytest
import numpy as np

import easyidp as idp


test_data = idp.data.TestData()


class TestBack2rawBatchConsistency:
    """Test that back2raw_batch produces identical results to back2raw."""

    @pytest.fixture
    def metashape_project(self):
        """Load the Lotus Metashape project."""
        return idp.Metashape(
            project_path=test_data.metashape.lotus_psx,
            chunk_id=0
        )

    @pytest.fixture
    def roi_with_z(self):
        """Load ROI and add Z values from DSM."""
        roi = idp.ROI(test_data.shp.lotus_shp, name_field=0)
        # Select a subset for testing
        key_list = list(roi.keys())
        for key in key_list:
            if key not in ["N1W1", "N1W2", "N1E1", "N1E2"]:
                del roi[key]
        roi.get_z_from_dsm(test_data.metashape.lotus_dsm)
        return roi

    def test_batch_consistency_with_original(
        self, metashape_project, roi_with_z
    ):
        """Verify batch version produces same results as original."""
        ms = metashape_project
        roi = roi_with_z

        # Run original method
        result_orig = ms.back2raw(roi)

        # Run batch method
        result_batch = ms.back2raw_batch(roi)

        # Compare structure
        assert set(result_orig.keys()) == set(result_batch.keys()), \
            "ROI keys mismatch"

        # Compare each ROI
        for roi_name in result_orig.keys():
            orig_photos = result_orig[roi_name]
            batch_photos = result_batch[roi_name]

            assert set(orig_photos.keys()) == set(batch_photos.keys()), \
                f"Photo keys mismatch for ROI [{roi_name}]"

            # Compare coordinates
            for photo_name in orig_photos.keys():
                orig_coords = orig_photos[photo_name]
                batch_coords = batch_photos[photo_name]

                np.testing.assert_array_almost_equal(
                    orig_coords,
                    batch_coords,
                    decimal=5,
                    err_msg=f"Coords mismatch for [{roi_name}][{photo_name}]"
                )


class TestBack2rawBatchPerformance:
    """Performance benchmarks comparing back2raw and back2raw_batch."""

    @pytest.fixture
    def metashape_project(self):
        """Load the Lotus Metashape project."""
        return idp.Metashape(
            project_path=test_data.metashape.lotus_psx,
            chunk_id=0
        )

    @pytest.fixture
    def roi_small(self):
        """Small ROI subset (4 plots)."""
        roi = idp.ROI(test_data.shp.lotus_shp, name_field=0)
        key_list = list(roi.keys())
        for key in key_list:
            if key not in ["N1W1", "N1W2", "N1E1", "N1E2"]:
                del roi[key]
        roi.get_z_from_dsm(test_data.metashape.lotus_dsm)
        return roi

    @pytest.fixture
    def roi_medium(self):
        """Medium ROI subset (16 plots)."""
        roi = idp.ROI(test_data.shp.lotus_shp, name_field=0)
        key_list = list(roi.keys())
        keep_keys = key_list[:16]
        for key in key_list:
            if key not in keep_keys:
                del roi[key]
        roi.get_z_from_dsm(test_data.metashape.lotus_dsm)
        return roi

    @pytest.fixture
    def roi_full(self):
        """Full ROI (all plots)."""
        roi = idp.ROI(test_data.shp.lotus_shp, name_field=0)
        roi.get_z_from_dsm(test_data.metashape.lotus_dsm)
        return roi

    def _benchmark(self, ms, roi, iterations: int = 3) -> dict:
        """
        Run benchmark for both methods.

        Parameters
        ----------
        ms : Metashape
            The Metashape project object.
        roi : ROI
            The ROI object to project.
        iterations : int
            Number of iterations for timing.

        Returns
        -------
        dict
            Benchmark results with times and speedup.
        """
        # Warm up
        _ = ms.back2raw(roi)
        _ = ms.back2raw_batch(roi)

        # Benchmark original
        times_orig = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = ms.back2raw(roi)
            times_orig.append(time.perf_counter() - start)

        # Benchmark batch
        times_batch = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = ms.back2raw_batch(roi)
            times_batch.append(time.perf_counter() - start)

        mean_orig = np.mean(times_orig)
        mean_batch = np.mean(times_batch)

        return {
            "original_mean": mean_orig,
            "original_std": np.std(times_orig),
            "batch_mean": mean_batch,
            "batch_std": np.std(times_batch),
            "speedup": mean_orig / mean_batch if mean_batch > 0 else 0,
            "n_rois": len(roi),
            "n_photos": len(ms.photos),
        }

    def test_performance_small_roi(self, metashape_project, roi_small):
        """Benchmark with small ROI (4 plots)."""
        result = self._benchmark(metashape_project, roi_small)

        print(f"\n{'='*60}")
        print(f"SMALL ROI Performance ({result['n_rois']} ROIs, "
              f"{result['n_photos']} photos)")
        print(f"{'='*60}")
        print(f"Original:  {result['original_mean']:.4f}s "
              f"± {result['original_std']:.4f}s")
        print(f"Batch:     {result['batch_mean']:.4f}s "
              f"± {result['batch_std']:.4f}s")
        print(f"Speedup:   {result['speedup']:.2f}x")
        print(f"{'='*60}")

        # Expect at least some speedup
        assert result["speedup"] >= 1.0, \
            "Batch version should not be slower than original"

    def test_performance_medium_roi(self, metashape_project, roi_medium):
        """Benchmark with medium ROI (16 plots)."""
        result = self._benchmark(metashape_project, roi_medium)

        print(f"\n{'='*60}")
        print(f"MEDIUM ROI Performance ({result['n_rois']} ROIs, "
              f"{result['n_photos']} photos)")
        print(f"{'='*60}")
        print(f"Original:  {result['original_mean']:.4f}s "
              f"± {result['original_std']:.4f}s")
        print(f"Batch:     {result['batch_mean']:.4f}s "
              f"± {result['batch_std']:.4f}s")
        print(f"Speedup:   {result['speedup']:.2f}x")
        print(f"{'='*60}")

        assert result["speedup"] >= 1.0

    def test_performance_full_roi(self, metashape_project, roi_full):
        """Benchmark with full ROI (all plots)."""
        result = self._benchmark(metashape_project, roi_full, iterations=2)

        print(f"\n{'='*60}")
        print(f"FULL ROI Performance ({result['n_rois']} ROIs, "
              f"{result['n_photos']} photos)")
        print(f"{'='*60}")
        print(f"Original:  {result['original_mean']:.4f}s "
              f"± {result['original_std']:.4f}s")
        print(f"Batch:     {result['batch_mean']:.4f}s "
              f"± {result['batch_std']:.4f}s")
        print(f"Speedup:   {result['speedup']:.2f}x")
        print(f"{'='*60}")

        assert result["speedup"] >= 1.0


class TestBack2rawBatchEdgeCases:
    """Test edge cases for back2raw_batch."""

    @pytest.mark.skip(reason="Requires full mocked Metashape project")
    def test_empty_photos(self):
        """Test with project that has no enabled photos."""
        # This test requires a fully mocked Metashape project with valid
        # transform matrix, which is complex to set up in unit tests.
        pass

    def test_disabled_chunk_raises(self):
        """Test that disabled chunk raises error."""
        ms = idp.Metashape(
            project_path=test_data.metashape.multichunk_psx,
            chunk_id=1
        )

        roi = idp.ROI()
        roi["test"] = np.array([[0, 0, 0], [1, 1, 1]])

        with pytest.raises(TypeError, match="Unable to process disabled chunk"):
            ms.back2raw_batch(roi)

    def test_2d_roi_raises(self):
        """Test that 2D ROI raises error."""
        ms = idp.Metashape(test_data.metashape.lotus_psx)

        roi = idp.ROI(test_data.shp.lotus_shp, name_field=0)
        # Don't add Z values - should raise error

        with pytest.raises(ValueError, match="requires 3D roi"):
            ms.back2raw_batch(roi)
