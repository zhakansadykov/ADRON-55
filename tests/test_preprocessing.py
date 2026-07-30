"""Unit tests for src/preprocessing.py.

Run with: pytest tests/test_preprocessing.py -v
"""
import numpy as np

from src import preprocessing

ALL_SECTIONS = [
    'left_g', 'right_g', 'front_g', 'back_g',
    'left_1', 'middle_1', 'right_1', 'front_1', 'back_1',
    'left_2', 'middle_2', 'right_2', 'front_2', 'back_2',
    'left_3', 'middle_3', 'right_3', 'front_3', 'back_3',
]


def _empty_event(**overrides):
    """A full set of zero-filled sections, with the given ones replaced."""
    arrays = {name: [0] * 72 for name in ALL_SECTIONS}
    arrays.update(overrides)
    return arrays


class TestPrepareArray:

    def test_truncates_at_minus_one(self):
        result = preprocessing.prepare_array([10, 20, 30, -1, 40, 50], 72)
        assert list(result) == [10, 20, 30]

    def test_no_minus_one(self):
        result = preprocessing.prepare_array([10, 20, 30], 72)
        assert list(result) == [10, 20, 30]

    def test_empty_array(self):
        assert len(preprocessing.prepare_array([], 72)) == 0


class TestPadToExpected:

    def test_pads_with_zeros(self):
        result = preprocessing.pad_to_expected(np.array([10, 20, 30]), 5)
        assert list(result) == [10, 20, 30, 0, 0]

    def test_truncates_if_longer(self):
        result = preprocessing.pad_to_expected(np.array([10, 20, 30, 40, 50, 60]), 4)
        assert list(result) == [10, 20, 30, 40]

    def test_no_change_if_equal(self):
        result = preprocessing.pad_to_expected(np.array([10, 20, 30]), 3)
        assert list(result) == [10, 20, 30]


class TestCombineOverlap:
    """Sections overlap: they are summed by index, never reversed or truncated."""

    def test_sums_by_index(self):
        result = preprocessing.combine_overlap([[1, 2, 3], [10, 20, 30]], 3)
        assert list(result) == [11, 22, 33]

    def test_does_not_reverse(self):
        """A signal in the first channel of both sections stays in channel 0."""
        result = preprocessing.combine_overlap([[100, 0, 0], [100, 0, 0]], 3)
        assert list(result) == [200, 0, 0]

    def test_short_section_does_not_truncate(self):
        """The shortest section must not clip the full-width ones."""
        result = preprocessing.combine_overlap([[1, 2, 3, 4, 5], [10, 20]], 5)
        assert list(result) == [11, 22, 3, 4, 5]

    def test_pads_to_expected_length(self):
        result = preprocessing.combine_overlap([[1, 2]], 5)
        assert list(result) == [1, 2, 0, 0, 0]


class TestUnifyLayers:

    def test_layer_ordering(self, config):
        """Four planes per projection, ordered bottom to top."""
        expected = config['geometry']['expected_channels']
        X, X_idx, Y, Y_idx = preprocessing.unify_layers(_empty_event(), expected)
        assert len(X) == 4 and len(Y) == 4
        assert X_idx == [6, 4, 2, 0]
        assert Y_idx == [7, 5, 3, 1]

    def test_plane_widths(self, config):
        """Each assembled plane has its nominal channel count."""
        expected = config['geometry']['expected_channels']
        X, X_idx, Y, Y_idx = preprocessing.unify_layers(_empty_event(), expected)
        for layer, idx in zip(X + Y, X_idx + Y_idx):
            assert len(layer) == expected[f'row_{idx + 1}']

    def test_basic_stitching(self, config):
        """Two parallel gamma sections are summed channel by channel."""
        expected = config['geometry']['expected_channels']
        arrays = _empty_event(left_g=[1, 2, 3, 4, 5, -1],
                              right_g=[10, 20, 30, 40, 50, -1])
        X, _, _, _ = preprocessing.unify_layers(arrays, expected)
        row_1 = X[3]                       # plane 1 is the last entry, bottom to top
        assert list(row_1[:5]) == [11, 22, 33, 44, 55]
        assert row_1[5:].sum() == 0

    def test_missing_section_is_zero_filled(self, config):
        """An absent section must not raise; it contributes zeros."""
        expected = config['geometry']['expected_channels']
        arrays = _empty_event(left_g=[7, 0, 0])
        del arrays['right_g']
        X, _, _, _ = preprocessing.unify_layers(arrays, expected)
        assert X[3][0] == 7


class TestChannelToMm:
    """Channel-to-position conversion and the per-plane alignment offsets."""

    def test_centres_by_own_plane_width(self, config):
        """The centre of a plane maps to zero when its offset is zero."""
        cfg = {'geometry': dict(config['geometry'], row_alignment_mm={})}
        n_ch = cfg['geometry']['expected_channels']['row_3']
        assert preprocessing.channel_to_mm(n_ch / 2.0, 2, cfg) == 0.0

    def test_pitch(self, config):
        """One channel corresponds to one pitch."""
        cfg = {'geometry': dict(config['geometry'], row_alignment_mm={})}
        w = cfg['geometry']['channel_width']
        n_ch = cfg['geometry']['expected_channels']['row_3']
        assert preprocessing.channel_to_mm(n_ch / 2.0 + 1, 2, cfg) == w

    def test_alignment_offset_is_subtracted(self, config):
        cfg = {'geometry': dict(config['geometry'], row_alignment_mm={2: -214.2})}
        n_ch = cfg['geometry']['expected_channels']['row_3']
        assert preprocessing.channel_to_mm(n_ch / 2.0, 2, cfg) == 214.2

    def test_planes_are_centred_independently(self, config):
        """Planes of different widths both map their own centre to zero."""
        cfg = {'geometry': dict(config['geometry'], row_alignment_mm={})}
        ch = cfg['geometry']['expected_channels']
        assert preprocessing.channel_to_mm(ch['row_3'] / 2.0, 2, cfg) == 0.0
        assert preprocessing.channel_to_mm(ch['row_4'] / 2.0, 3, cfg) == 0.0
