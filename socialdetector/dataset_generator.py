import os
import numpy as np
from PIL import Image

from tensorflow.python.data import Dataset
import tensorflow as tf
from tensorflow.python.data.experimental import sample_from_datasets
from tensorflow.python.data.experimental.ops.matching_files import MatchingFilesDataset
from tensorflow.python.framework.tensor_shape import Dimension
from tensorflow_datasets.core import benchmark

from socialdetector.dct_utils import blockwise_dct_matrix, coefficient_order
from socialdetector.utility import imread_mode


def lazily(ds: Dataset):
    return Dataset.from_generator(lambda: ds, output_types=ds.output_types, output_shapes=ds.output_shapes)


def path_dataset(key: str, patterns):
    return MatchingFilesDataset(patterns).map(lambda x: {key: x})
    # return Dataset.list_files(patterns).map(lambda x: {key: x})


def dataset_apply(parent: Dataset, transformer: callable, src_key: str, dst_key: str, flat=False):
    if flat:
        return parent.flat_map(
            lambda entry: Dataset.from_tensor_slices(transformer(entry[src_key])).map(lambda x: {**entry, dst_key: x}))
    else:
        def m(entry):
            entry[dst_key] = transformer(entry[src_key])
            return entry

        return parent.map(m)


def path_bind(dst_path: str, origin_dir: str):
    def apply(path):
        path = path.decode("utf-8")
        dir_path, name = os.path.split(os.path.abspath(path))
        relative_dir_path = os.path.relpath(dir_path, origin_dir)
        return os.path.join(dst_path, relative_dir_path, name)

    return lambda path: tf.numpy_function(apply, [path], tf.string)


def path_append(append: str):
    return lambda path: path + append


def load_image(mode="RGB", dtype=np.uint8):
    return lambda x: tf.numpy_function(lambda path: imread_mode(path, mode, dtype), [x], dtype)


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


def split_image_fn(tile_size, strides=None):
    t = [1, *tile_size, 1]
    r = [1 for _ in t]
    if strides is None:
        s = t
    else:
        s = [1, *strides, 1]

    def split(image):
        patches = tf.image.extract_patches(image, sizes=t, strides=s, rates=r, padding='VALID')
        return tf.reshape(patches[0], (-1, *tile_size, *image.shape[3:]))

    return split


def reshape_block_dct(considered_coefficients=10):
    considered_coefficients = coefficient_order[:considered_coefficients]

    def apply(x):
        shape = [tf.shape(x)[k] for k in range(4)]
        return tf.gather(tf.reshape(x, (*shape[:-2], 64)), considered_coefficients, axis=-1)[tf.newaxis, ...]

    return apply


def encode_coefficients(considered_coefficients=10):
    def apply(x):
        x = tf.reshape(x, (-1, considered_coefficients))
        x = tf.einsum("i...j->j...i", x)
        x = tf.map_fn(lambda a: tf.histogram_fixed_width(a, (-50.5, 50.5), nbins=101, dtype=tf.int32), x,
                      fn_output_signature=tf.int32)
        x = tf.reshape(x, (-1,))
        return x

    return apply


def add_properties(properties):
    def apply(x):
        x.update(properties)
        return x

    return apply


def log_message(fn):
    def apply(x):
        print(fn(x))
        return x

    return apply


def full_dataset(path_ds, noiseprint_path, origin_path, considered_coefficients=9, src_key='origin_file',
                 dst_key='output', shuffle=200, log_path=None):
    ds = path_ds
    ds = dataset_apply(ds, path_bind(noiseprint_path, origin_path), src_key, 'noiseprint_file')
    ds = dataset_apply(ds, path_append('.npz'), src_key='noiseprint_file', dst_key='noiseprint_file')
    ds = dataset_apply(ds, load_image('YCbCr'), src_key, 'ycbcr_image')
    ds = dataset_apply(ds, block_dct(), 'ycbcr_image', 'block_dct')
    ds = dataset_apply(ds, reshape_block_dct(considered_coefficients), src_key='block_dct', dst_key='block_dct')
    ds = dataset_apply(ds, split_image_fn((8, 8), strides=(7, 7)), src_key='block_dct', dst_key='block_dct_split',
                       flat=True)
    ds = dataset_apply(ds, encode_coefficients(considered_coefficients), src_key='block_dct_split', dst_key=dst_key)
    if shuffle:
        ds = ds.shuffle(shuffle)
    return ds


def datasets_concatenate(datasets):
    ds = datasets[0]
    for d in datasets[1:]:
        ds = ds.concatenate(d)
    return ds


def datasets_interleave(datasets, block_length=None, cycle_length=None):
    datasets = tuple(datasets)
    if cycle_length is not None:
        return datasets_concatenate(
            [datasets_interleave(datasets[i:i + cycle_length], block_length=block_length) for i in
             range(0, len(datasets), cycle_length)])

    choices = Dataset.range(len(datasets))
    if block_length is not None:
        choices = choices.flat_map(lambda x: Dataset.from_tensors(x).repeat(block_length))

    return tf.data.experimental.choose_from_datasets(datasets, choices.cache().repeat())


def my_data_generator(shuffle=1000, batch_size=64, seed=0, validate=100, test=100):
    if isinstance(batch_size, int):
        batch_size = tuple(batch_size for _ in range(3))

    def my_datasets(folder, label):
        key_input = 'input'
        key_output = 'output'
        path_ds = path_dataset(key_input, folder)
        path_ds = path_ds.shuffle(100000, seed=seed)

        val = path_ds.take(validate).cache()
        tst = path_ds.skip(validate).take(test).cache()
        trn = path_ds.skip(validate+test)

        v_files = list(val.map(lambda x: x[key_input]).as_numpy_iterator())
        t_files = list(tst.map(lambda x: x[key_input]).as_numpy_iterator())

        print("Validation files: %d" % len(v_files))
        print(v_files)
        print("Test files: %d" % len(t_files))
        print(t_files)

        def generator(p_ds):
            ds = full_dataset(p_ds,
                              "C:\\Users\\franz\\Downloads\\Datasets\\ucid_social\\ucid_noiseprint",
                              "C:\\Users\\franz\\Downloads\\Datasets\\ucid_social\\ucid", shuffle=shuffle,
                              src_key=key_input, dst_key=key_output)
            ds = ds.map(lambda x: x[key_output])
            return Dataset.zip((ds, Dataset.from_tensors(label).repeat()))

        return generator(trn), generator(val), generator(tst)

    facebook = my_datasets("C:\\Users\\franz\\Downloads\\Datasets\\ucid_social\\ucid\\facebook\\**\\*.jpg", [1, 0, 0])
    flickr = my_datasets("C:\\Users\\franz\\Downloads\\Datasets\\ucid_social\\ucid\\flickr\\**\\*.jpg", [0, 1, 0])
    twitter = my_datasets("C:\\Users\\franz\\Downloads\\Datasets\\ucid_social\\ucid\\twitter\\**\\*.jpg", [0, 0, 1])

    output = []
    for i in range(3):
        interleaved = datasets_interleave((facebook[i], flickr[i], twitter[i]))
        if batch_size[i] > 0:
            interleaved = interleaved.batch(batch_size[i])
        output.append(interleaved)
    return output
