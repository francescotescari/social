import os

from tensorflow.python.data import Dataset
import tensorflow as tf
import numpy as np
from socialdetector.dataset.social_images.social_images import SocialImages, stride_to_str
from socialdetector.dataset_generator import reshape_block_dct
from socialdetector.dataset_utils import datasets_concatenate, datasets_interleave


class DsSplit:
    tile_size = (64, 64)

    @staticmethod
    def count_chunks(ds):
        total = 0
        for entry in ds.as_numpy_iterator():
            total += entry['chunks']
        return total

    @staticmethod
    def steps_until_chunks(ds, target):
        stop_index = []
        total = 0
        for entry in ds.as_numpy_iterator():
            chunks = entry['chunks']
            total += chunks
            if total >= target:
                chunks = chunks - (total - target)
                stop_index.append((chunks, entry['path']))
                break
            stop_index.append((chunks, entry['path']))
        return stop_index

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
        self.mappings = None
        self.labels_map = None
        self.block_tile_size = [s // 8 for s in self.tile_size]
        self.val_ds = None
        self.tst_ds = None

        self.encode_fn = None
        if self.dct_encoding is not None:
            reshape = reshape_block_dct(self.considered_coefficients)
            encode = self.dct_encoding(self.considered_coefficients)
            re = tf.function(lambda x: encode(reshape(x)),
                             input_signature=(tf.TensorSpec((*self.block_tile_size, 64), dtype=tf.float16),))
            if self.noiseprint:
                self.encode_fn = lambda d, n: (re(d), n)
            else:
                self.encode_fn = re

    def to_final_dataset(self, label, dataset: Dataset, shuffle: int, max_size=None, max_chunks_per_image=None):
        flat_fn = self.get_chunks(max_chunks_per_image)
        ds = dataset.flat_map(flat_fn)
        if self.encode_fn is not None:
            ds = ds.map(self.encode_fn, num_parallel_calls=self.parallel,
                        deterministic=(self.deterministic and shuffle > 0))
        if shuffle > 0:
            ds = ds.shuffle(shuffle, seed=self.seed, reshuffle_each_iteration=True)
        if max_size is not None:
            ds = ds.take(max_size)
        ds = self.labelize(label, ds)
        return ds

    def labelize(self, k, v):
        lb_ds = Dataset.from_tensors(self.labels_map[k]).repeat()
        return Dataset.zip((v, lb_ds))

    @property
    def labels(self):
        im = {tuple(v): k for k, v in self.labels_map.items()}
        n = len(im)

        def label_of(i, n):
            zeros = [0] * n
            zeros[i] = 1
            return zeros

        return [im[tuple(label_of(i, n))] for i in range(n)]

    @property
    def min_chunks(self):
        if self._min_chunks is None:
            raise ValueError("Not split yet")
        return self._min_chunks

    def get_chunks(self, max_chunks):
        max_chunks = None if max_chunks is None else int(max_chunks)
        getter = (lambda x: x) if max_chunks is None else (lambda x: x[:max_chunks])

        def process_entry(path, block_strides):
            if isinstance(path, np.ndarray):
                path = path[()]
            path = path.decode("utf-8")
            data_path = "%s.%s.npz" % (path, stride_to_str(block_strides))
            data = np.load(data_path)
            dct_patches = data['dct']
            noiseprint_patches = data['noiseprint']
            assert len(dct_patches) == len(noiseprint_patches)
            # print(path, len(dct_patches), max_chunks)
            return getter(dct_patches), getter(noiseprint_patches)

        def apply(x):
            dct, noise = tf.numpy_function(process_entry, (x['path'], x['strides']), (tf.float16, tf.float16))
            # tf.print(tf.shape(dct), max_chunks)
            dct.set_shape((None, *self.block_tile_size, 64))
            noise.set_shape((None, *self.tile_size))
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

        self.labels_map = labels_map


        chunks_length = {k: self.count_chunks(ds) for k, ds in datasets.items()}
        min_chunks = min(chunks_length.values())
        min_images = min([len(ds) for ds in datasets.values()])
        validation_size = min_images // 10
        test_size = min_images // 10

        self._min_chunks = round((min_chunks * 0.8) * len(chunks_length))
        print({'test_images': test_size, 'val_images': validation_size,
               'train_images': (min_images - test_size - validation_size)})
        # shuffle
        shuffled = {k: ds.shuffle(len(ds), seed=self.seed, reshuffle_each_iteration=False).cache() for k, ds in
                    datasets.items()}
        # cache the shuffled paths
        paths = [list(ds) for ds in shuffled.values()]

        def take_images(ds_set, image_size, same_chunks=True):
            ds = {k: ds.take(image_size).cache() for k, ds in ds_set.items()}
            chunks_limit = min([self.count_chunks(d) for d in ds.values()]) if same_chunks else None
            remain = {k: ds.skip(image_size) for k, ds in ds_set.items()}
            max_chunks_per_image = {k: None for k in ds}
            mappings = {}
            for label, dataset in ds.items():
                max_chunks = None
                if chunks_limit is not None:
                    sorted_chunks = sorted(dataset.map(lambda x: x['chunks']).as_numpy_iterator())
                    remaining_images = image_size
                    total = chunks_limit
                    max_chunks = None
                    for num_chunks in sorted_chunks:
                        if total <= num_chunks * remaining_images:
                            max_chunks = np.ceil(total / remaining_images)
                            break
                        total -= num_chunks
                        remaining_images -= 1

                    max_chunks_per_image[label] = max_chunks
                if max_chunks is None:
                    it = dataset.map(lambda x: (x['path'], x['chunks'])).as_numpy_iterator()
                else:
                    it = dataset.map(lambda x: (x['path'], tf.minimum(x['chunks'], max_chunks))).as_numpy_iterator()
                if chunks_limit is None:
                    res = list(it)
                else:
                    res = []
                    total = chunks_limit
                    for entry in it:
                        chunks = entry[1]
                        if total < chunks:
                            res.append((entry[0], total))
                            break

                        res.append(entry)
                        total -= chunks
                mappings[label] = res

            return ds, mappings, remain, max_chunks_per_image, chunks_limit

        val_ds, val_map, shuffled, max_chunks_val, val_chunks_limit = take_images(shuffled, validation_size)
        tst_ds, tst_map, shuffled, max_chunks_tst, tst_chunks_limit = take_images(shuffled, test_size, False)
        self.val_ds = val_ds
        self.tst_ds = tst_ds
        self.mappings = (None, val_map, tst_map)

        def filenames_of(ds):
            return list(map(lambda x: os.path.basename(x['path'].decode("utf-8")), ds.as_numpy_iterator()))

        print("Validation", {k: filenames_of(v) for k, v in val_ds.items()})
        print("Test", {k: filenames_of(v) for k, v in tst_ds.items()})

        train_ds = {k: ds.shuffle(len(ds), seed=self.seed, reshuffle_each_iteration=True).cache() for k, ds in
                    shuffled.items()}

        validation_ds = [self.to_final_dataset(k, ds, 0, val_chunks_limit, max_chunks_val[k]) for k, ds in
                         val_ds.items()]
        validation_ds = datasets_concatenate(validation_ds).prefetch(self.parallel)

        test_ds = [self.to_final_dataset(k, ds, 0, tst_chunks_limit, max_chunks_tst[k]) for k, ds in tst_ds.items()]
        test_ds = datasets_concatenate(test_ds).prefetch(self.parallel)

        train_ds = [self.to_final_dataset(k, ds, self.shuffle_train) for k, ds in train_ds.items()]
        train_ds = [ds.repeat() for ds in train_ds]
        train_ds = datasets_interleave(train_ds).prefetch(self.parallel)

        return train_ds, validation_ds, test_ds
