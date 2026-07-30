"""Shared pytest fixtures.

The configuration fixture loads the committed template, so the tests exercise
the same parameter values as the published analysis.
"""
import os

import pytest
import yaml

from src.event_quality import Cluster, TrackCandidate

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'config', 'settings.yaml.example',
)


@pytest.fixture(scope='session')
def config():
    """The released configuration template."""
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


@pytest.fixture
def mock_track():
    """Build a TrackCandidate with a given total energy and channel slope.

    The track carries one single-channel cluster in each of the three
    hadron-block planes of its projection, so that it is a well-formed
    three-point candidate. The energy is split evenly between the clusters,
    and the centroid advances by `slope` channels per millimetre of height.
    """
    x_planes = [7, 5, 3]   # plane numbers, bottom to top
    y_planes = [8, 6, 4]
    heights = (4275, 4200, 1460, 1300, 1010, 820, 630, 470)

    def _make(energy=1000.0, slope=0.0, projection='X', track_id=0,
              start_channel=20.0):
        planes = x_planes if projection == 'X' else y_planes
        per_hit = float(energy) / len(planes)
        z0 = heights[planes[0] - 1]
        hits = []
        for plane in planes:
            layer_idx = plane - 1
            z = heights[layer_idx]
            channel = int(round(start_channel + slope * (z - z0)))
            hits.append(Cluster(
                layer_idx=layer_idx,
                projection=projection,
                channels=[channel],
                amplitudes=[per_hit],
            ))
        return TrackCandidate(
            track_id=track_id,
            seed_layer=planes[0],
            projection=projection,
            hits=hits,
            slope=slope,
            penetration_depth=len(hits),
        )

    return _make
