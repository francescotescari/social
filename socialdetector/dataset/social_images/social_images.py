"""social_images dataset."""
import random
from abc import ABC
from collections import defaultdict

import skimage
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image
from tensorflow.python.data import Dataset
from tensorflow_datasets.core import splits as splits_lib, split_builder as split_builder_lib

from socialdetector.dct_utils import blockwise_dct_matrix
from socialdetector.experiment import DatasetSplitter
from socialdetector.noiseprint import NoiseprintEngineV2

from socialdetector.utility import *

_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(social_images): BibTeX citation
_CITATION = """
"""


def extract_patches(im, tile_size, strides):
    window = [*tile_size, *im.shape[len(tile_size):]]
    strides = [1 if i >= len(strides) else strides[i] for i in range(len(im.shape))]
    patches = skimage.util.view_as_windows(im, window, strides)
    return np.reshape(patches, [-1, *window])


def map_features_type(feature):
    if isinstance(feature, tfds.features.Tensor):
        return feature.dtype
    elif isinstance(feature, tfds.features.Text):
        return tf.string
    raise ValueError()


def map_features_shape(feature):
    if isinstance(feature, tfds.features.Tensor):
        return feature.shape
    elif isinstance(feature, tfds.features.Text):
        return ()
    raise ValueError()


class DataSplitter(ABC):

    def split(self, path_generators) -> dict:
        raise NotImplementedError()


class SocialImages(tfds.core.GeneratorBasedBuilder, ABC):
    """DatasetBuilder for social_images dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    labels = None

    def __init__(self, url, rel_input_dirs, name, tile_size=(64, 64)):
        self.labels = sorted(set(entry[1] for entry in rel_input_dirs))
        self.url = url
        self.rel_input_dirs = rel_input_dirs
        self.name = name
        self.tile_size = tile_size
        self.block_tile_size = [t // 8 for t in tile_size]
        super().__init__()

    def features(self):
        return {
            'label': tfds.features.Text(),
            'path': tfds.features.Text(),
        }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(self.features()),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('dct', 'label'),  # e.g. ('image', 'label')
            homepage='https://dataset-homepage/',
            citation=_CITATION,
        )

    def split_gen(self, *args, **kwargs):
        return self._split_generators(*args, **kwargs)

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        # download and extract dataset
        dataset_dir = dl_manager.download_and_extract(self.url)

        # group glob patterns by label
        label_paths = defaultdict(list)
        for (path, label) in self.rel_input_dirs:
            generator = path_glob_iterator(dataset_dir, path)
            label_paths[label].append(generator)

        # interleave path generators
        label_generators = {k: interleave_generators(v) if len(v) > 1 else v[0] for k, v in label_paths.items()}
        label_generators = {k: list(map(lambda x: (x, k), v)) for k, v in label_generators.items()}

        engine = NoiseprintEngineV2()

        return {k: self._generate_examples(v, engine) for k, v in label_generators.items()}

    def _generate_examples(self, dataset, engine) -> split_builder_lib.SplitGenerator:
        cached_requests = defaultdict(list)

        for (path, label) in dataset:
            image = Image.open(path)
            quality = jpeg_quality_of(image)
            image = np.asarray(image.convert('YCbCr'))[..., 0]

            quality_request = cached_requests[quality]
            quality_request.append((path, image, label))

            if len(quality_request) < 32:
                continue

            cached_requests[quality] = []
            for request in quality_request:
                yield self._process_image(request[0], request[1], quality, request[2], engine)

        for quality, quality_request in cached_requests.items():
            for request in quality_request:
                yield self._process_image(request[0], request[1], quality, request[2], engine)

    def _process_image(self, path: str, image: np.ndarray, quality: int, label: str, engine: NoiseprintEngineV2):
        data_path = "%s.data.npz" % path
        if not os.path.exists(data_path):
            block_dct = blockwise_dct_matrix(image.astype(np.int16) - 128, add_padding=False)
            block_dct = block_dct.reshape((*block_dct.shape[:-2], 64)).astype(np.float16)
            engine.load_session(quality)
            noiseprint = engine.predict(image.astype(np.float32) / 256.0).astype(np.float16)
            dct_patches = extract_patches(block_dct, self.block_tile_size, self.block_tile_size)
            noiseprint_patches = extract_patches(noiseprint, self.tile_size, self.tile_size)
            np.savez(data_path, dct=dct_patches, noiseprint=noiseprint_patches)
        return path, {
            'path': path,
            'label': label
        }


class ImageSplitter(tfds.core.GeneratorBasedBuilder, ABC):
    """DatasetBuilder for social_images dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    url = None
    rel_input_dirs = ()
    splitter: DatasetSplitter
    immediate = False

    def __init__(self, data_dir=".", immediate=False, tile_size=(64, 64)):
        self.tile_size = tile_size
        self.block_tile_size = [s // 8 for s in tile_size]
        self.dct_shape = (*self.block_tile_size, 64)
        self.generator = SocialImages(url=self.url, rel_input_dirs=self.rel_input_dirs, name=self.name + "_plain")
        self.labels = self.generator.labels
        self._dl_man = None
        self.immediate = immediate
        self._split_data = None
        super().__init__()

    def features(self):
        return {
            'dct': tfds.features.Tensor(shape=self.dct_shape, dtype=tf.float16),
            'label': tfds.features.ClassLabel(names=self.labels),
            'path': tfds.features.Text(),
            'part': tfds.features.Tensor(shape=[], dtype=tf.int32),
            'noiseprint': tfds.features.Tensor(shape=(*self.tile_size, 1), dtype=tf.float16)
        }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(self.features()),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('dct', 'label'),  # e.g. ('image', 'label')
            homepage='https://dataset-homepage/',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager = None):
        if self._split_data is None:
            self.generator.download_and_prepare()
            generators = self.generator.as_dataset().values()
            generators = map(lambda x: x.as_numpy_iterator(), generators)

            generator = [map(lambda x: {k: v.decode("utf-8") for k, v in x.items()}, gen) for gen in generators]

            self._split_data = self.splitter.split(generator)

        return {k: self._generate_examples(v) for k, v in self._split_data.items()}

    def _generate_examples(self, dataset) -> split_builder_lib.SplitGenerator:
        for entry in dataset:
            for generated in self._process_entry(entry):
                yield generated

    def _process_entry(self, entry):
        path = entry['path']
        data_path = "%s.data.npz" % path
        data = np.load(data_path)
        dct_patches = data['dct']
        noiseprint_patches = data['noiseprint']
        assert len(dct_patches) == len(noiseprint_patches)
        for i in range(len(dct_patches)):
            new_entry = {**entry}
            new_entry['dct'] = dct_patches[i]
            new_entry['noiseprint'] = noiseprint_patches[i][..., np.newaxis]
            new_entry['part'] = i
            yield "%s.part%d" % (path, i), new_entry

    def download_and_prepare(self, *, download_dir=None, download_config=None):
        if self.immediate:
            self.generator.download_and_prepare()
            config = download_config or tfds.download.DownloadConfig()
            self._dl_man = self._make_download_manager(download_dir=download_dir, download_config=config)
        else:
            super().download_and_prepare(download_dir=download_dir, download_config=download_config)

    def as_dataset(self, *args, **kwargs):
        if not self.immediate:
            return super(ImageSplitter, self).as_dataset(*args, **kwargs)
        else:
            kwargs.update(zip(super().as_dataset.__code__.co_varnames[1:], args))
            generators = self._split_generators(self._dl_man)
            split = None
            for key in kwargs:
                if key == 'split':
                    split = kwargs['split']
                else:
                    raise NotImplementedError("option %s not supported in immediate mode" % key)
            if split is not None:
                generators = {split: generators[split]}
            out_types = {k: map_features_type(v) for k, v in self.features().items()}
            out_shapes = {k: map_features_shape(v) for k, v in self.features().items()}
            res = {}
            labels = self.labels
            label_map = {labels[i]: i for i in range(len(labels))}

            def indexer(entry):
                entry['label'] = label_map[entry['label']]
                return entry

            def get_generator(split_name):
                gen = self._split_generators(self._dl_man)[split_name]
                gen = map(lambda x: x[1], gen)
                gen = map(indexer, gen)
                return gen

            for split_name in generators:
                ds = Dataset.from_generator(lambda x=split_name: get_generator(split_name), output_types=out_types,
                                            output_shapes=out_shapes).cache()
                res[split_name] = ds
            return res if split is None else res[split]


class FixedSizeSplitter(DataSplitter):

    def __init__(self, test_size, validation_size, seed=None):
        self.test_size = test_size
        self.validation_size = validation_size
        self.seed = seed

    def split(self, path_generators) -> dict:
        path_lists = [list(gen) for gen in path_generators]
        if self.seed is not None:
            random.seed(self.seed)
        [random.shuffle(l) for l in path_lists]
        whole_files = list(interleave_generators([iter(l) for l in path_lists]))
        total = len(whole_files)
        test_limit = round(total * self.test_size / 100)
        validation_limit = round(total * self.validation_size / 100)
        res = {
            'train': whole_files[(test_limit + validation_limit):]
        }
        if test_limit > 0:
            res['test'] = whole_files[:test_limit]
        if validation_limit > 0:
            res['validation'] = whole_files[test_limit:(test_limit + validation_limit)]
        return res


class UcidPublic(ImageSplitter):
    url = "http://lci.micc.unifi.it/labd/datasets/public.zip"
    rel_input_dirs = (
        ("facebook\\**\\*.jpg", 'facebook'),
        ("flickr\\**\\*.jpg", 'flickr'),
        ("twitter\\**\\*.jpg", 'twitter')
    )

    splitter = FixedSizeSplitter(10, 10, 12321)


class UcidSocial(ImageSplitter):
    url = "http://lci.micc.unifi.it/labd/datasets/ucid.zip"
    rel_input_dirs = (
        ("facebook\\**\\*.jpg", 'facebook'),
        ("flickr\\**\\*.jpg", 'flickr'),
        ("twitter\\**\\*.jpg", 'twitter')
    )
    splitter = FixedSizeSplitter(10, 10, 12321)
