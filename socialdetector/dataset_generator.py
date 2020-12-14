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


def block_dct():
    def apply(x):
        tensor = tf.numpy_function(
            lambda y_cb_cr_data: blockwise_dct_matrix((y_cb_cr_data[..., 0]).astype(np.int16) - 128), [x], tf.double)
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
    ds = path_ds.cache()
    np_ds = None
    dc_ds = None
    if noiseprint:
        np_ds = ds.map(path_bind(noiseprint_path, origin_path)).map(path_append(".npz")).map(load_noiseprint())
        split = split_image_fn(tile_size, strides)
        np_ds = np_ds.flat_map(lambda e: Dataset.from_tensor_slices(split(e)))
    if dct:
        dc_ds = ds.map(load_image('YCbCr')).map(block_dct()).map(reshape_block_dct(considered_coefficients))
        split = split_image_fn(block_tile_size, block_strides)
        dc_ds = dc_ds.flat_map(lambda e: Dataset.from_tensor_slices(split(e)))
        dc_ds = dc_ds.map(dct_encoding(considered_coefficients))

    datasets = tuple(d for d in (np_ds, dc_ds) if d is not None)
    return Dataset.zip(datasets) if len(datasets) > 1 else datasets[0]


def my_data_generator(class_dirs, validate, test, shuffle, batch_size, seed, print_val_tst_set=True,
                      noiseprint_path=None,
                      dct_encoding=None, **kwargs):
    if not is_listing(batch_size):
        batch_size = tuple(batch_size for _ in range(3))

    def my_datasets(folder, label, g_seed):
        path_ds = MatchingFilesDataset(folder)
        path_ds = path_ds.shuffle(10000000, seed=seed, reshuffle_each_iteration=False)  # Load all paths shuffled

        val = path_ds.take(validate).cache()
        tst = path_ds.skip(validate).take(test).cache()
        trn = path_ds.skip(validate + test).cache().shuffle(1000, reshuffle_each_iteration=True)

        if print_val_tst_set:
            print("Validation files of %s:" % folder)
            print_ds(val)
            print("Test files of %s:" % folder)
            print_ds(tst)

        def generator(p_ds):
            ds = full_dataset(p_ds,
                              noiseprint_path=noiseprint_path,
                              dct_encoding=dct_encoding,
                              **kwargs)
            ds = ds.shuffle(shuffle, seed=seed + 37 + g_seed * 13, reshuffle_each_iteration=True)
            return Dataset.zip((ds, Dataset.from_tensors(label).repeat()))

        return generator(trn), generator(val), generator(tst)

    def label_of(i, n):
        lb = [0 for _ in range(n)]
        lb[i] = 1
        return lb

    n = len(class_dirs)
    class_dss = [my_datasets(class_dirs[i], label_of(i, n), i) for i in range(n)]

    output = []
    for i in range(3):
        interleaved = datasets_interleave([class_ds[i] for class_ds in class_dss])
        if batch_size[i] > 0:
            interleaved = interleaved.batch(batch_size[i])
        output.append(interleaved)
    return output


default_seed = 12321


class Splittable:
    estimated_len = None

    def split_build(self, batch_size=(256, 256, 0), encoding=None, noiseprint=False, **additional_configs):
        raise NotImplementedError

    @staticmethod
    def merge(splittables, ratios=1):

        def split_build(*args, **kwargs):
            trains = []
            vals = []
            tests = []
            for split in splittables:
                t, v, ts = split.split_build(*args, **kwargs)
                trains.append(t)
                vals.append(v)
                tests.append(ts)
            return datasets_interleave(trains, ratios), datasets_interleave(vals, ratios), datasets_interleave(tests, ratios)

        res = lambda : None
        res.split_build = split_build
        return res





class DatasetSpec(Splittable):

    def __init__(self, class_dirs, noiseprint_dir, origin_dir, default_validation, default_test):
        self.class_dirs = class_dirs
        self.noiseprint_path = noiseprint_dir
        self.origin_path = origin_dir
        self.default_validation = default_validation
        self.default_test = default_test

    def split_build(self, batch_size=(256, 256, 0), encoding=None, noiseprint=False, **additional_configs):
        config = {
            'class_dirs': self.class_dirs,
            'seed': default_seed,
            'validate': self.default_validation,
            'test': self.default_test,
            'batch_size': batch_size,
            'dct_encoding': encoding,
            'shuffle': 5000
        }
        if noiseprint:
            config['noiseprint_path'] = self.noiseprint_path
            config['origin_path'] = self.origin_path
        config.update(additional_configs)

        train, validation, test = my_data_generator(**config)
        validation = validation.cache()
        test = test.cache()
        train = train

        return train, validation, test
