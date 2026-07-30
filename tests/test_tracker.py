"""Unit tests for the projection matching of src/tracker.py.

Run with: pytest tests/test_tracker.py -v
"""
from src import tracker


class TestProjectionMatching:
    """Pairing of X and Y track candidates into three-dimensional tracks."""

    def test_energy_matching_low_ratio(self, mock_track, config):
        """An energy ratio below the threshold is matched by energy rank."""
        x = mock_track(energy=1000, projection='X')
        y = mock_track(energy=1200, projection='Y')
        matches = tracker.match_xy_tracks_hybrid([x], [y], config)
        assert len(matches) == 1
        assert matches[0][2] == 'energy'

    def test_geometry_matching_high_ratio(self, mock_track, config):
        """An energy ratio above the threshold falls back to geometry."""
        x = mock_track(energy=1000, projection='X')
        y = mock_track(energy=5000, projection='Y')
        matches = tracker.match_xy_tracks_hybrid([x], [y], config)
        assert len(matches) == 1
        assert matches[0][2] == 'geometry'

    def test_no_tracks(self, config):
        """Empty input gives an empty result."""
        assert tracker.match_xy_tracks_hybrid([], [], config) == []

    def test_only_x_tracks(self, mock_track, config):
        """A projection without a partner gives no match."""
        x = mock_track(energy=1000, projection='X')
        assert tracker.match_xy_tracks_hybrid([x], [], config) == []

    def test_multiple_tracks(self, mock_track, config):
        """Several tracks are paired in order of energy."""
        x1 = mock_track(energy=1000, slope=0.001, projection='X', track_id=0)
        x2 = mock_track(energy=500, slope=0.002, projection='X', track_id=1)
        y1 = mock_track(energy=1100, slope=0.001, projection='Y', track_id=0)
        y2 = mock_track(energy=550, slope=0.002, projection='Y', track_id=1)

        matches = tracker.match_xy_tracks_hybrid([x1, x2], [y1, y2], config)
        assert len(matches) == 2

        energies_x = sorted((tracker.compute_track_energy(m[0]) for m in matches), reverse=True)
        energies_y = sorted((tracker.compute_track_energy(m[1]) for m in matches), reverse=True)
        assert energies_x == [1000, 500]
        assert energies_y == [1100, 550]


class TestTrackEnergy:
    """Summed cluster amplitude of a track."""

    def test_compute_energy(self, mock_track):
        assert tracker.compute_track_energy(mock_track(energy=1000)) == 1000


class TestFittingConfig:
    """The 3D-fit parameters are read from the configuration, not hard-coded."""

    def test_reads_config(self, config):
        sigma, chi2_max, min_hits = tracker._fitting_cfg(config)
        assert sigma == config['tracking']['fitting']['sigma_typical_mm']
        assert chi2_max == config['tracking']['fitting']['chi2_ndf_threshold']
        assert min_hits == config['tracking']['fitting']['min_hits_3d']

    def test_falls_back_to_defaults(self):
        sigma, chi2_max, min_hits = tracker._fitting_cfg({})
        assert (sigma, chi2_max, min_hits) == (200.0, 10.0, 3)


class TestGammaPlaneExclusion:
    """The gamma planes must not enter the direction fit."""

    def test_excluded_indices(self, config):
        assert tracker.fit_exclude_idx(config) == {0, 1}
