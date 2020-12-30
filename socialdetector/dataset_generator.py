import os
import sys

import numpy as np
from PIL import Image
from tensorflow.python.data import Dataset
import tensorflow as tf
from tensorflow.python.data.experimental.ops.matching_files import MatchingFilesDataset
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.models import Model
from tensorflow.python.ops.confusion_matrix import confusion_matrix

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
        if len(x.shape) > 3:
            shape = [tf.shape(x)[k] for k in range(len(x.shape))]
            x = tf.reshape(x, (*shape[:-2], 64))
        x = tf.cast(x, tf.float32)
        return tf.gather(x, considered_coefficients, axis=-1)[tf.newaxis, ...]

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
        np_ds = ds \
            .map(path_bind(noiseprint_path, origin_path), **map_config) \
            .map(path_append(".npz"), **map_config) \
            .map(load_noiseprint(), **map_config)
        split = split_image_fn(tile_size, strides)
        np_ds = np_ds.flat_map(lambda e: Dataset.from_tensor_slices(split(e)))
    if dct:
        dc_ds = ds \
            .map(load_image('YCbCr'), **map_config) \
            .map(block_dct(), **map_config) \
            .map(reshape_block_dct(considered_coefficients), **map_config)
        split = split_image_fn(block_tile_size, block_strides)
        dc_ds = dc_ds.flat_map(lambda e: Dataset.from_tensor_slices(split(e)))
        dc_ds = dc_ds.map(dct_encoding(considered_coefficients), **map_config)

    datasets = tuple(d for d in (np_ds, dc_ds) if d is not None)
    return Dataset.zip(datasets) if len(datasets) > 1 else datasets[0]


default_seed = 12321


def apply_to_patches(slide, overlap, padding, apply_fn):
    side = slide + 2 * overlap

    def apply(x):
        ow = x.shape[0]
        oh = x.shape[1]

        rounder = tf.math.ceil

        w = rounder(ow / slide) * slide
        h = rounder(oh / slide) * slide
        nw, nh = w // slide, h // slide

        x = tf.pad(x, [[overlap, overlap + (w - ow)], [overlap, overlap + (h - oh)]], mode=padding)[
            tf.newaxis, ..., tf.newaxis]
        patches = tf.image.extract_patches(x, [1, side, side, 1], [1, slide, slide, 1], [1, 1, 1, 1], 'VALID')
        patches = tf.reshape(patches, (-1, side, side, 1))

        generated = apply_fn(patches)
        rec = tf.slice(generated, [0, overlap, overlap, 0], [-1, slide, slide, 1])
        rec = tf.reshape(rec, [1, nw, nh, slide * slide])
        rec = tf.nn.depth_to_space(rec, slide, data_format="NHWC")
        rec = tf.squeeze(rec)
        rec = tf.slice(rec, [0, 0], [ow, oh])
        return rec

    return apply


def tdfs_encode(dataset: Dataset, labels=3, noiseprint=False, dct_encoding=None,
                parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False,
                considered_coefficients=9, shuffle=10000, seed=None):
    reshape = reshape_block_dct(considered_coefficients=considered_coefficients)

    def label_of(x):
        # return x['label']
        return tf.one_hot(x['label'], depth=labels)

    if dct_encoding is not None:
        dct_encoding = dct_encoding(considered_coefficients)
        if noiseprint:
            apply = lambda x: ((dct_encoding(reshape(x['dct'])), x['noiseprint']), label_of(x))
        else:
            apply = lambda x: (dct_encoding(reshape(x['dct'])), label_of(x))
        ds = dataset.map(apply, num_parallel_calls=parallel_calls, deterministic=deterministic)
    else:
        ds = dataset.map(lambda x: (x['noiseprint'], label_of(x)), num_parallel_calls=parallel_calls,
                         deterministic=deterministic)
    ds = ds.prefetch(parallel_calls)
    if shuffle > 0:
        ds = ds.shuffle(shuffle, reshuffle_each_iteration=True, seed=seed)
    return ds


class Metrics(Callback):

    def __init__(self, val_generator: Dataset):
        super().__init__()

        def filter_fn(i, n):
            zeros = [0] * n
            zeros[i] = 1
            return lambda x, y: tf.reduce_all(y == zeros)

        label_shape = val_generator.output_shapes[1]
        n = label_shape[0]

        self.cls = [val_generator.filter(filter_fn(i, n)).batch(128).cache() for i in range(n)]

    def on_epoch_end(self, epoch, logs=None):
        self.model: Model
        # print("L3", len(self.cls3))
        hand = getattr(self.model, "_eval_data_handler", None)
        self.model._eval_data_handler = None
        for i in range(len(self.cls)):
            print("Class %d:" % i)
            self.model.evaluate(self.cls[i])
        self.model._eval_data_handler = hand


class ParSine(tf.keras.layers.Layer):
    def __init__(self, w0: float = 1.0, **kwargs):
        """
        Sine activation function with w0 scaling support.

        Args:
            w0: w0 in the activation step `act(x; w0) = sin(w0 * x)`
        """
        super(ParSine, self).__init__(**kwargs)
        self.w0 = w0
        self.w1 = 1.0
        self.p = 0.5

    def call(self, inputs, **kwargs):
        return self.p * tf.sin(self.w1 * inputs + self.w0) + (1 - self.p) * inputs

    def get_config(self):
        config = {'w0': self.w0, 'w1': self.w1, 'p': self.p}
        base_config = super(ParSine, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
