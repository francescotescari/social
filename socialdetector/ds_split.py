from tensorflow.python.data import Dataset
import tensorflow as tf
import numpy as np
from socialdetector.dataset.social_images.social_images import SocialImages, stride_to_str
from socialdetector.dataset_generator import reshape_block_dct
from socialdetector.dataset_utils import datasets_concatenate, datasets_interleave


class DsSplit:

    @staticmethod
    def count_chunks(ds):
        total = 0
        for entry in ds.as_numpy_iterator():
            total += entry['chunks']
        return total

    @staticmethod
    def steps_until_chunks(ds, target):
        total = 0
        i = 0
        for entry in ds.as_numpy_iterator():
            total += entry['chunks']
            i += 1
            if total >= target:
                break
        return i

    def __init__(self, noiseprint=False, dct_encoding=None, seed=123, shuffle_train=2000, considered_coefficients=9,
                 parallel=tf.data.experimental.AUTOTUNE, deterministic=False):
        self.noiseprint = noiseprint
        self.dct_encoding = dct_encoding
        self.seed = seed
        self.shuffle_train = shuffle_train
        self.considered_coefficients = considered_coefficients
        self.parallel = parallel
        self.deterministic = deterministic
        self._min_chunks = None

    @property
    def min_chunks(self):
        if self._min_chunks is None:
            raise ValueError("Not split yet")
        return self._min_chunks

    def get_chunks(self, tile_size, block_tile_size):

        def process_entry(path, block_strides):
            path = path.decode("utf-8")
            data_path = "%s.%s.npz" % (path, stride_to_str(block_strides))
            data = np.load(data_path)
            dct_patches = data['dct']
            noiseprint_patches = data['noiseprint']
            assert len(dct_patches) == len(noiseprint_patches)
            return dct_patches, noiseprint_patches

        def apply(x):
            dct, noise = tf.numpy_function(process_entry, (x['path'], x['strides']), (tf.float16, tf.float16))
            dct.set_shape((None, *block_tile_size, 64))
            noise.set_shape((None, *tile_size))
            dss = []
            if self.dct_encoding is not None:
                dss.append(Dataset.from_tensor_slices(dct))
            if self.noiseprint:
                dss.append(Dataset.from_tensor_slices(noise))
            return dss[0] if len(dss) < 2 else Dataset.zip(tuple(dss))

        return apply

    def split_datasets(self, ds_builder: SocialImages):
        datasets = ds_builder.as_dataset()
        labels = sorted(set(datasets.keys()))
        labels_map = {}
        for i in range(len(labels)):
            zeros = [0] * len(labels)
            zeros[i] = 1
            labels_map[labels[i]] = zeros

        print(labels_map)

        def labelize(k, v):
            lb_ds = Dataset.from_tensors(labels_map[k]).repeat()
            return Dataset.zip((v, lb_ds))

        flat_fn = self.get_chunks(ds_builder.tile_size, ds_builder.block_tile_size)
        encode_fn = lambda x: x
        if self.dct_encoding is not None:
            reshape = reshape_block_dct(self.considered_coefficients)
            encode = self.dct_encoding(self.considered_coefficients)
            re = tf.function(lambda x: encode(reshape(x)),
                             input_signature=(tf.TensorSpec((*ds_builder.block_tile_size, 64), dtype=tf.float16),))
            if self.noiseprint:
                e = lambda d, n: (re(d), n)
            else:
                e = re
            encode_fn = lambda x: x.map(e, num_parallel_calls=self.parallel, deterministic=self.deterministic)

        def to_final_dataset(label, dataset: Dataset, shuffle: int, max_size=None):
            ds = dataset.flat_map(flat_fn)
            ds = encode_fn(ds)
            if shuffle > 0:
                ds = ds.shuffle(shuffle, seed=self.seed, reshuffle_each_iteration=True)
            if max_size is not None:
                ds = ds.take(max_size)
            ds = labelize(label, ds)
            return ds

        chunks_length = {k: self.count_chunks(ds) for k, ds in datasets.items()}
        min_chunks = min(chunks_length.values())
        validation_size = min_chunks // 10
        test_size = min_chunks // 10
        self._min_chunks = (min_chunks - validation_size - test_size)*len(chunks_length)
        print({'test_chunks': test_size, 'val_chunks': test_size, 'train_chunks': self._min_chunks})
        # shuffle
        shuffled = {k: ds.shuffle(len(ds), seed=self.seed, reshuffle_each_iteration=False).cache() for k, ds in
                    datasets.items()}
        # cache the shuffled paths
        [list(ds) for ds in shuffled.values()]
        validation_images = {k: self.steps_until_chunks(ds, validation_size) for k, ds in shuffled.items()}
        validation_ds = {k: ds.take(validation_images[k]).cache() for k, ds in shuffled.items()}
        shuffled = {k: ds.skip(validation_images[k]) for k, ds in shuffled.items()}
        test_images = {k: self.steps_until_chunks(ds, test_size) for k, ds in shuffled.items()}
        test_ds = {k: ds.take(test_images[k]).cache() for k, ds in shuffled.items()}
        shuffled = {k: ds.skip(test_images[k]).cache() for k, ds in shuffled.items()}
        train_ds = {k: ds.shuffle(len(ds), seed=self.seed, reshuffle_each_iteration=True).cache() for k, ds in
                    shuffled.items()}

        validation_ds = [to_final_dataset(k, ds, 100, validation_size) for k, ds in validation_ds.items()]
        validation_ds = datasets_concatenate(validation_ds).prefetch(self.parallel)

        test_ds = [to_final_dataset(k, ds, 100, test_size) for k, ds in test_ds.items()]
        test_ds = datasets_concatenate(test_ds).prefetch(self.parallel)

        train_ds = [to_final_dataset(k, ds, self.shuffle_train) for k, ds in train_ds.items()]
        train_ds = [ds.repeat() for ds in train_ds]
        train_ds = datasets_interleave(train_ds).prefetch(self.parallel)

        return train_ds, validation_ds, test_ds
