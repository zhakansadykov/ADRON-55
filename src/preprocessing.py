import numpy as np
import logging
logger = logging.getLogger(__name__)


def prepare_array(arr, target_len):
    """Truncate the array at the first -1 marker."""
    arr = np.array(arr, dtype=np.float64)
    neg1_idx = np.where(arr == -1)[0]
    if len(neg1_idx) > 0:
        arr = arr[:neg1_idx[0]]
    # Padding is applied later, in combine_overlap
    return arr


def pad_to_expected(data, expected_len):
    """Pad the array with zeros, or truncate it, to the expected length."""
    if len(data) < expected_len:
        return np.pad(data, (0, expected_len - len(data)), 
                     mode='constant', constant_values=0)
    return data[:expected_len]


def combine_overlap(cameras_data, expected_len):
    """Sum the sections of one plane, aligned BY ARRAY INDEX (overlap).

    Per the detector channel map, element[0] of every section of a plane maps to
    the same transverse position (left_g[0]<->right_g[0],
    right_1[0]<->middle_1[0]<->left_1[0]), so the sections are combined WITHOUT
    reversal. Each section is padded with zeros, or truncated, to expected_len;
    it is never truncated to the length of the shortest section.
    A particle crosses one section at a given depth, so at each transverse
    position usually only one term is non-zero.
    """
    total = np.zeros(expected_len, dtype=np.float64)
    for arr in cameras_data:
        total = total + pad_to_expected(np.asarray(arr, dtype=np.float64), expected_len)
    return total


def unify_layers(arrays_dict, expected_channels):
    """
    Assemble the raw sections into the X and Y planes (index-aligned overlap).

    The parallel sections of a plane sit at different depths but cover the same
    transverse positions (see the detector channel map). Assembly is therefore a
    sum of the sections aligned by array index, WITHOUT reversal and WITHOUT
    truncation to the shortest one. Absent sections contribute zeros.

    expected_channels: plane widths (row_1..row_8): 50/69/48/72/48/72/48/72.
    """
    all_camera_keys = [
        'left_g', 'right_g', 'front_g', 'back_g',
        'left_1', 'middle_1', 'right_1', 'front_1', 'back_1',
        'left_2', 'middle_2', 'right_2', 'front_2', 'back_2',
        'left_3', 'middle_3', 'right_3', 'front_3', 'back_3',
    ]
    prepared = {}
    for k in all_camera_keys:
        if k in arrays_dict:
            prepared[k] = prepare_array(arrays_dict[k], 72)
        else:
            prepared[k] = np.zeros(0, dtype=np.float64)
            logger.debug(f"Section '{k}' absent from the event; filled with zeros.")

    # Sections of each plane, aligned by index. Suffixes: _g = planes 1-2,
    # _1 = planes 3-4, _2 = planes 5-6, _3 = planes 7-8.
    row_1 = combine_overlap([prepared['left_g'],  prepared['right_g']],                         expected_channels['row_1'])
    row_2 = combine_overlap([prepared['front_g'], prepared['back_g']],                          expected_channels['row_2'])
    row_3 = combine_overlap([prepared['right_1'], prepared['middle_1'], prepared['left_1']],    expected_channels['row_3'])
    row_4 = combine_overlap([prepared['front_1'], prepared['back_1']],                          expected_channels['row_4'])
    row_5 = combine_overlap([prepared['right_2'], prepared['middle_2'], prepared['left_2']],    expected_channels['row_5'])
    row_6 = combine_overlap([prepared['front_2'], prepared['back_2']],                          expected_channels['row_6'])
    row_7 = combine_overlap([prepared['right_3'], prepared['middle_3'], prepared['left_3']],    expected_channels['row_7'])
    row_8 = combine_overlap([prepared['front_3'], prepared['back_3']],                          expected_channels['row_8'])

    # Planes for the track finder, ordered BOTTOM TO TOP, with their
    # plane indices (0 = top .. 7 = bottom)
    X_layers = [row_7, row_5, row_3, row_1]
    X_idx = [6, 4, 2, 0]
    Y_layers = [row_8, row_6, row_4, row_2]
    Y_idx = [7, 5, 3, 1]
    return X_layers, X_idx, Y_layers, Y_idx

def channel_to_mm(centroid: float, layer_idx: int, cfg: dict) -> float:
    """
    Convert a channel position (a centroid within its own plane) to millimetres
    relative to the detector axis, applying the data-driven plane alignment.

    The plane is centred by its own width, and the measured plane offset
    (row_alignment_mm) is subtracted. That offset follows from a physical
    constraint: for an azimuthally symmetric cosmic-ray flux the mean track
    position in a plane coincides with its geometric centre. Removing it
    eliminates the large per-plane displacements (dead regions, mechanical
    misalignment) that would otherwise impose a fixed tilt and bias the azimuth.
    """
    geom = cfg['geometry']
    cw = geom['channel_width']
    expected = geom.get('expected_channels', {})
    n_ch = expected.get(f'row_{layer_idx + 1}', geom.get('max_channels', 72))
    base = (centroid - n_ch / 2.0) * cw

    # Plane alignment offset (YAML keys may be int or str)
    align = geom.get('row_alignment_mm', {}) or {}
    offset = align.get(layer_idx, align.get(str(layer_idx), 0.0))
    return base - offset
