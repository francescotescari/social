import tensorflow as tf
from tensorflow.python.data import Dataset

from socialdetector.utility import is_listing, glob_iterator, path_glob_iterator


def print_ds(ds):
    print(list(ds.as_numpy_iterator()))


def datasets_concatenate(datasets):
    ds = datasets[0]
    for d in datasets[1:]:
        ds = ds.concatenate(d)
    return ds


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


def tf_print(*args):
    tf.print(*args)
    return args[0]


def str_endswith(end_regex):
    rg = ".*" + end_regex + "$"
    return lambda x: tf.strings.regex_full_match(x, rg, name="tf_ends_with")


def datasets_interleave(datasets, block_length=None, cycle_length=None):
    datasets = tuple(datasets)
    if cycle_length is not None:
        return datasets_concatenate(
            [datasets_interleave(datasets[i:i + cycle_length], block_length=block_length) for i in
             range(0, len(datasets), cycle_length)])

    choices = Dataset.range(len(datasets))

    if block_length is not None:
        if not is_listing(block_length):
            block_length = tuple(block_length for _ in range(len(datasets)))
        choices = datasets_concatenate(
            [Dataset.from_tensors(tf.convert_to_tensor(i, dtype=tf.int64)).repeat(block_length[i]) for i in
             range(len(datasets))])

    return tf.data.experimental.choose_from_datasets(datasets, choices.cache().repeat())


def glob_dataset(pattern, recursive=True):
    return Dataset.from_generator(lambda: glob_iterator(pattern, recursive=recursive), output_types=tf.string)


def path_glob_dataset(path, pattern, recursive=True):
    return Dataset.from_generator(lambda: path_glob_iterator(path, pattern, recursive=recursive),
                                  output_types=tf.string)
