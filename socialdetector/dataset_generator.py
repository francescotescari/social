import os
import numpy as np
from PIL import Image
from tensorflow.python.data import Dataset
import tensorflow as tf
from tensorflow.python.data.experimental.ops.matching_files import MatchingFilesDataset

from socialdetector.dataset_utils import *
from socialdetector.dct_utils import blockwise_dct_matrix, coefficient_order
from socialdetector.utility import imread_mode, is_listing


def get_quantization_table():
    def apply(path: str):
        try:
            return Image.open(path).quantization
        except AttributeError:
            return None

    return lambda path: tf.numpy_function(apply, [path], None)


def block_dct(add_padding=False):
    def apply(x):
        tensor = tf.numpy_function(
            lambda y_cb_cr_data: blockwise_dct_matrix((y_cb_cr_data[..., 0]).astype(np.int16) - 128,
                                                      add_padding=add_padding), [x], tf.double)
        tensor.set_shape((None, None, 8, 8))
        return tensor

    return apply


def reshape_block_dct(considered_coefficients=10):
    considered_coefficients = coefficient_order[:considered_coefficients]

    def apply(x):
        shape = [tf.shape(x)[k] for k in range(4)]
        return tf.gather(tf.reshape(x, (*shape[:-2], 64)), considered_coefficients, axis=-1)[tf.newaxis, ...]

    return apply


def encode_coefficients_paper(considered_coefficients=10):
    def apply(x):
        x = tf.reshape(x, (-1, considered_coefficients))
        x = tf.einsum("i...j->j...i", x)
        x = tf.clip_by_value(x, -50, 50)
        size = x.shape[-1]
        x = tf.map_fn(lambda a: tf.histogram_fixed_width(a, (-50.5, 50.5), nbins=101, dtype=tf.int32), x,
                      fn_output_signature=tf.int32)
        x = tf.reshape(x, (-1,)) / size

        return x[..., tf.newaxis]

    return apply


def encode_coefficients_my(considered_coefficients=10):
    def apply(x):
        x = tf.reshape(x, (-1, considered_coefficients))
        x = tf.einsum("i...j->j...i", x)
        size = x.shape[-1]
        results = []
        for i in range(1, 21):
            tmp = x / i
            diff = tmp - tf.round(tmp)
            results.append([tf.histogram_fixed_width(diff[j], (-0.5, 0.5), 11) for j in range(considered_coefficients)])

        x = tf.convert_to_tensor(results)
        x = tf.einsum("ij...->ji...", x)

        return x / size

    return apply


def load_noiseprint():
    def apply(path):
        data = np.load(path)
        return data[next(iter(data.files))]

    def apply_ts(entry):
        ten = tf.numpy_function(apply, (entry,), Tout=tf.float16)
        ten.set_shape((None, None))
        return tf.cast(ten, tf.float64)[tf.newaxis, ..., tf.newaxis]

    return apply_ts


def full_dataset(path_ds, dct_encoding, noiseprint_path, origin_path=None, considered_coefficients=9,
                 tile_size=(64, 64), strides=(56, 56)):
    noiseprint = noiseprint_path is not None
    dct = dct_encoding is not None
    if not noiseprint and not dct:
        raise ValueError("Please specify either the noiseprint_path (and origin_path) or the dct_encoding")
    if origin_path is None and noiseprint:
        raise ValueError(
            "Please specify the origin path to relativize the path entries and find the final noiseprint path")
    block_tile_size = [round(s / 8) for s in tile_size]
    block_strides = [round(s / 8) for s in strides]
    ds = path_ds
    np_ds = None
    dc_ds = None
    map_config = {
        'deterministic': False,
        'num_parallel_calls': tf.data.experimental.AUTOTUNE
    }
    if noiseprint:
        np_ds = ds\
            .map(path_bind(noiseprint_path, origin_path), **map_config)\
            .map(path_append(".npz"), **map_config)\
            .map(load_noiseprint(), **map_config)
        split = split_image_fn(tile_size, strides)
        np_ds = np_ds.flat_map(lambda e: Dataset.from_tensor_slices(split(e)))
    if dct:
        dc_ds = ds\
            .map(load_image('YCbCr'), **map_config)\
            .map(block_dct(), **map_config)\
            .map(reshape_block_dct(considered_coefficients), **map_config)
        split = split_image_fn(block_tile_size, block_strides)
        dc_ds = dc_ds.flat_map(lambda e: Dataset.from_tensor_slices(split(e)))
        dc_ds = dc_ds.map(dct_encoding(considered_coefficients), **map_config)

    datasets = tuple(d for d in (np_ds, dc_ds) if d is not None)
    return Dataset.zip(datasets) if len(datasets) > 1 else datasets[0]




default_seed = 12321





