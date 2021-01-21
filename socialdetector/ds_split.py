import math
import os
from typing import Any

from tensorflow.python.data import Dataset
import tensorflow as tf
import numpy as np
from socialdetector.dataset.social_images.social_images import SocialImages, stride_to_str
from socialdetector.dataset_generator import reshape_block_dct
from socialdetector.dataset_utils import datasets_concatenate, datasets_interleave, split_image_fn
from socialdetector.utility import is_listing


class DsSplit:
    tile_size = (64, 64)

    def chunks_of(self, shape, strides):
        on_x = (shape[0] - self.tile_size[0]) // strides[0] + 1
        on_y = (shape[1] - self.tile_size[1]) // strides[1] + 1
        return on_x * on_y

    def count_chunks(self, ds, strides=None):
        if strides is None:
            strides = self.tile_size
        total = 0
        for entry in ds.as_numpy_iterator():
            total += self.chunks_of(entry['shape'], strides)
        return total

    def steps_until_chunks(self, ds, target, strides=None):
        if strides is None:
            strides = self.tile_size
        stop_index = []
        total = 0
        for entry in ds.as_numpy_iterator():
            chunks = self.chunks_of(entry['shape'], strides)
            total += chunks
            if total >= target:
                chunks = chunks - (total - target)
                stop_index.append((chunks, entry['path']))
                break
            stop_index.append((chunks, entry['path']))
        return stop_index

    def __init__(self, noiseprint=False, dct_encoding=None, seed=123, shuffle_train=2000, considered_coefficients=9,
                 parallel=tf.data.experimental.AUTOTUNE, deterministic=False, debug=False):
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
        self.class_weights = None
        self.debug = debug

    def labelize(self, k, v: Dataset, deterministic, sw=None):
        def mk_getter(n):
            if callable(n):
                return n
            if is_listing(n):
                return lambda args: tuple(args[i] for i in n)
            else:
                if n is not None:
                    return lambda args: args[n]
                else:
                    return lambda args: tuple()

        lb = self.labels_map[k]
        if sw is not None:
            sw = mk_getter(sw)
            mfn = lambda *args: (args[0], lb, sw(args[1]))
        else:
            mfn = lambda *args: (args[0], lb)

        return v.map(mfn, num_parallel_calls=self.parallel, deterministic=deterministic)


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

    def _load_data(self, path):
        data_path = "%s.data.npz" % path.decode("utf-8")
        data = np.load(data_path)
        return data['dct'], data['noiseprint']

    def _load_data_tf(self, path):
        dct, noiseprint = tf.numpy_function(self._load_data, (path,), (tf.float16, tf.float16), name='load_data_tf')
        dct.set_shape((None, None, 64))
        noiseprint.set_shape((None, None))
        return dct, noiseprint

    def _split_image_fn(self, block_strides=None, max_chunks=None):
        if block_strides is None:
            block_strides = self.block_tile_size
        strides = [8 * s for s in block_strides]
        split_dct = split_image_fn(self.block_tile_size, block_strides)
        split_noiseprint = split_image_fn(self.tile_size, strides)
        apply = None

        if max_chunks is None:
            taker = lambda x: x
        else:
            taker = lambda x: x.shuffle(len(x), seed=self.seed).take(max_chunks)

        def ret(ds, *args):
            args = (len(ds), *args)
            args = Dataset.from_tensors(args).repeat()
            ds = taker(ds)
            return Dataset.zip((ds, args))

        if self.noiseprint and self.dct_encoding is not None:
            def apply(dct, noiseprint, *args):
                dct_chunks = split_dct(dct[tf.newaxis, ...])
                noiseprint_chunks = split_noiseprint(noiseprint[tf.newaxis, ..., tf.newaxis])
                # tf.print("ACT", tf.shape(noiseprint), tf.shape(dct_chunks)[0])
                return ret(Dataset.zip(
                    (Dataset.from_tensor_slices(dct_chunks), Dataset.from_tensor_slices(noiseprint_chunks))), *args)
        elif self.noiseprint:
            def apply(dct, noiseprint, *args):
                return ret(Dataset.from_tensor_slices(split_noiseprint(noiseprint[tf.newaxis, ..., tf.newaxis])), *args)
        elif self.dct_encoding:
            def apply(dct, noiseprint, *args):
                return ret(Dataset.from_tensor_slices(split_dct(dct[tf.newaxis, ...])), *args)
        assert apply is not None
        return apply

    def _encode_fn(self, more_data):
        def wrap(fn):
            if more_data:
                return lambda dn, db: (fn(dn), db)
            else:
                return lambda dn, db: fn(dn)

        if self.dct_encoding is not None:
            reshape = reshape_block_dct(self.considered_coefficients)
            encode = self.dct_encoding(self.considered_coefficients)
            tf_encode_dct = tf.function(lambda x: encode(reshape(x)),
                                        input_signature=(tf.TensorSpec((*self.block_tile_size, 64), dtype=tf.float16),))
            if self.noiseprint:
                return wrap(lambda dn: (tf_encode_dct(dn[0]), dn[1]))
            else:
                return wrap(tf_encode_dct)
        else:
            assert self.noiseprint
            return wrap(lambda x: x)

    def augemntation_strides(self, train_chunks):
        max_chunks = max(train_chunks.values())
        max_data_augmentation = {k: max_chunks / v for k, v in train_chunks.items()}
        aug_strides = {}
        for label, aug in max_data_augmentation.items():
            axis_aug = math.sqrt(aug)
            strides = [max(1, math.ceil(s / axis_aug)) for s in self.block_tile_size]
            aug_strides[label] = strides
        return aug_strides

    def take_images(self, ds_set, image_size, same_chunks=False, strides=None):
        if strides is None:
            strides = self.tile_size
        ds = {k: ds.take(image_size).cache() for k, ds in ds_set.items()}
        chunks_limit = min([self.count_chunks(d) for d in ds.values()]) if same_chunks else None
        remain = {k: ds.skip(image_size) for k, ds in ds_set.items()}
        max_chunks_per_image = {k: None for k in ds}
        mappings = {}
        for label, dataset in ds.items():
            max_chunks = None
            ch_ds = [(x['path'], self.chunks_of(x['shape'], strides)) for x in dataset.as_numpy_iterator()]
            if chunks_limit is not None:
                sorted_chunks = sorted(map(lambda x: x[1], ch_ds))
                remaining_images = image_size
                total = chunks_limit
                max_chunks = None
                for num_chunks in sorted_chunks:
                    if total < num_chunks * remaining_images:
                        max_chunks = np.ceil(total / remaining_images)
                        break
                    total -= num_chunks
                    remaining_images -= 1

                max_chunks_per_image[label] = max_chunks
            if max_chunks is not None:
                ch_ds = map(lambda x: (x[0], min(x[1], max_chunks)), ch_ds)
            if chunks_limit is None:
                res = list(ch_ds)
            else:
                res = []
                total = chunks_limit
                for entry in ch_ds:
                    chunks = entry[1]
                    if total < chunks:
                        res.append((entry[0], total))
                        break

                    res.append(entry)
                    total -= chunks
            mappings[label] = res

        return ds, mappings, remain, max_chunks_per_image, chunks_limit

    def split_datasets(self, ds_builder):
        datasets = ds_builder.as_dataset()
        labels = sorted(set(datasets.keys()))
        labels_map = {}
        for i in range(len(labels)):
            zeros = [0] * len(labels)
            zeros[i] = 1
            labels_map[labels[i]] = zeros

        self.labels_map = labels_map

        chunks_length = {k: self.count_chunks(ds) for k, ds in datasets.items()}

        images = {k: len(ds) for k, ds in datasets.items()}
        avg_chunks = {k: chunks_length[k] / images[k] for k in chunks_length}

        min_images = min(images.values())
        validation_size = min_images // 10
        test_size = min_images // 10
        train_images = {k: n - validation_size - test_size for k, n in images.items()}

        print({'test_images': test_size, 'val_images': validation_size,
               'train_images': (min_images - test_size - validation_size)})
        print('Chunks:', chunks_length)
        # shuffle
        shuffled = {k: ds.shuffle(len(ds), seed=self.seed, reshuffle_each_iteration=False).cache() for k, ds in
                    datasets.items()}
        # cache the shuffled paths
        paths = [list(ds) for ds in shuffled.values()]

        val_ds, val_map, shuffled, max_chunks_val, val_chunks_limit = self.take_images(shuffled, validation_size)
        tst_ds, tst_map, shuffled, max_chunks_tst, tst_chunks_limit = self.take_images(shuffled, test_size)
        val_chunks = {k: self.count_chunks(ds) for k, ds in val_ds.items()}
        tst_chunks = {k: self.count_chunks(ds) for k, ds in val_ds.items()}
        print("Val chunks", val_chunks)
        print("Tst chunks", tst_chunks)
        print("Max chunks per image val", max_chunks_val, val_chunks_limit)
        print("Max chunks per image tst", max_chunks_tst, tst_chunks_limit)
        train_chunks = {k: self.count_chunks(ds) for k, ds in shuffled.items()}
        print("Train chunks", train_chunks)

        augmentation_strides = {k: self.block_tile_size for k in shuffled}
        # augmentation_strides = self.augemntation_strides(train_chunks)
        print("Aug strides", augmentation_strides)
        after_aug_train = {k: self.count_chunks(ds, [8 * s for s in augmentation_strides[k]]) for k, ds in
                           shuffled.items()}
        self._min_chunks = sum(after_aug_train.values())
        print("After aug", after_aug_train)

        max_chunks = max(after_aug_train.values())
        class_weights = {k: max_chunks / v for k, v in after_aug_train.items()}
        print("Class weights", class_weights)
        self.class_weights = {i: class_weights[self.labels[i]] for i in range(len(class_weights))}
        # self.class_weights = None
        256 / len(class_weights)
        inter_block = {k: round(256 / len(class_weights) / class_weights[k]) for k in class_weights}
        print("Inter block", inter_block)

        self.val_ds = val_ds
        self.tst_ds = tst_ds
        self.mappings = (None, val_map, tst_map)

        def filenames_of(ds):
            return list(map(lambda x: os.path.basename(x['path'].decode("utf-8")), ds.as_numpy_iterator()))

        print("Validation", {k: filenames_of(v) for k, v in val_ds.items()})
        print("Test", {k: filenames_of(v) for k, v in tst_ds.items()})

        train_ds = {k: ds.cache().shuffle(len(ds), seed=self.seed, reshuffle_each_iteration=True) for k, ds in
                    shuffled.items()}

        avg_val_chunks = np.mean(list(val_chunks.values()))
        print('Avg chunk in val', avg_val_chunks)
        val_sw = lambda x: avg_val_chunks/validation_size/x[0]
        validation_ds = [self.to_final_dataset(k, ds, 0, val_chunks_limit, max_chunks_val[k], sample_weight=val_sw) for
                         k, ds in
                         val_ds.items()]
        validation_ds = datasets_concatenate(validation_ds).prefetch(self.parallel)

        test_ds = [self.to_final_dataset(k, ds, 0, tst_chunks_limit, max_chunks_tst[k]) for k, ds in tst_ds.items()]
        test_ds = datasets_concatenate(test_ds).prefetch(self.parallel)


        sample_weight = 'default'
        sample_weight = {k: lambda x, n=(after_aug_train[k]/train_images[k]): (n/x[0]) for k in after_aug_train }
        # sample_weight = {k: lambda x, n=after_aug_train[k]/train_images[k]: n/x[1][0] for k in after_aug_train}
        if self.debug:
            sample_weight = lambda x: x
        print("sample_weight", sample_weight, self.debug)

        interleave_before = False

        if not isinstance(sample_weight, dict):
            sample_weight = {k: sample_weight for k in train_ds}
        train_ds = [self.to_final_dataset(k, ds, -1 if interleave_before else self.shuffle_train // len(self.labels),
                                          block_strides=augmentation_strides[k],
                                          sample_weight=sample_weight[k]) for k, ds in train_ds.items()]

        if interleave_before:
            block_len = tuple(inter_block.values())
            block_len = None
            train_ds = datasets_interleave(train_ds, block_length=block_len).repeat()
            train_ds = train_ds.shuffle(self.shuffle_train, reshuffle_each_iteration=True, seed=self.seed)
            self.class_weights = None
        else:
            train_ds = [ds.repeat() for ds in train_ds]
            train_ds = datasets_interleave(train_ds)
            self.class_weights = None

        train_ds = train_ds.prefetch(self.shuffle_train)

        return train_ds, validation_ds, test_ds

    def to_final_dataset(self, label, dataset: Dataset, shuffle: int, max_size=None, max_chunks_per_image=None,
                         block_strides=None, sample_weight: Any = 'default', repeat=False):
        if sample_weight == 'default':
            if max_chunks_per_image is None:
                sample_weight = lambda x: 1000 / x[0]
            else:
                sample_weight = lambda x: max_chunks_per_image / tf.reduce_min([x[1][0], max_chunks_per_image])

        deterministic = self.deterministic or shuffle == 0
        flat_fn = self._split_image_fn(block_strides=block_strides, max_chunks=max_chunks_per_image)
        encode = self._encode_fn(sample_weight is not None or self.debug)
        ds = dataset
        # ds = dat_dbg(ds, lambda x: tf.print("PRED", x['shape'], self.chunks_of(x['shape'], [s*8 for s in (block_strides if block_strides is not None else self.block_tile_size)])))
        ds = ds.map(lambda x: x['path'], num_parallel_calls=self.parallel, deterministic=deterministic)
        ds = ds.map(lambda x: (*self._load_data_tf(x), x), num_parallel_calls=self.parallel,
                    deterministic=deterministic)
        ds = ds.flat_map(flat_fn)
        if encode is not None:
            ds = ds.map(encode, num_parallel_calls=self.parallel, deterministic=deterministic)
        if repeat:
            ds = ds.repeat()
        if shuffle > 0:
            ds = ds.shuffle(shuffle, seed=self.seed, reshuffle_each_iteration=True)
        if max_size is not None:
            ds = ds.take(max_size)
        ds = self.labelize(label, ds, sw=sample_weight, deterministic=deterministic)
        return ds
