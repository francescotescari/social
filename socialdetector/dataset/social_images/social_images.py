"""social_images dataset."""
import math
import random
import shutil
from abc import ABC
from collections import defaultdict

import imagesize
import skimage
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image
from tensorflow.python.data import Dataset
from tensorflow_datasets.core import splits as splits_lib, split_builder as split_builder_lib

from socialdetector.dct_utils import blockwise_dct_matrix
from socialdetector.noiseprint import NoiseprintEngineV2, normalize_noiseprint

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


def stride_to_str(strides):
    return "strides_%d_%d" % (strides[0], strides[1])


def open_image_quality(path):
    image = Image.open(path)
    quality = jpeg_quality_of(image)
    image = np.asarray(image.convert('YCbCr'))[..., 0]
    return image, quality


def compute_block_dct(image):
    block_dct = blockwise_dct_matrix(image.astype(np.int16) - 128, add_padding=False)
    return block_dct.reshape((*block_dct.shape[:-2], 64))


def compute_noiseprint(image, quality, engine=None):
    if engine is None:
        engine = NoiseprintEngineV2()
    engine.load_session(quality)
    return normalize_noiseprint(engine.predict(image.astype(np.float32) / 256.0))


class SocialImages(tfds.core.GeneratorBasedBuilder, ABC):
    """DatasetBuilder for social_images dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    labels = None
    rel_input_dirs = ()
    url = None

    def __init__(self, data_dir=".", tile_size=(64, 64)):
        self.labels = sorted(set(entry[1] for entry in self.rel_input_dirs))
        self.tile_size = tile_size
        self.block_tile_size = [t // 8 for t in tile_size]
        self.strides = {}
        super().__init__()

    def features(self):
        return {
            'path': tfds.features.Text(),
            'shape': tfds.features.Tensor(shape=(2,), dtype=tf.int32)
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
            supervised_keys=None,  # e.g. ('image', 'label')
            homepage='https://dataset-homepage/',
            citation=_CITATION,
        )

    def split_gen(self, *args, **kwargs):
        return self._split_generators(*args, **kwargs)

    def _on_after_download(self, dir):
        pass

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        # download and extract dataset
        dataset_dir = dl_manager.download_and_extract(self.url)

        self._on_after_download(dataset_dir)

        # group glob patterns by label
        label_paths = defaultdict(list)
        for (path, label) in self.rel_input_dirs:
            generator = path_glob_iterator(dataset_dir, path)
            label_paths[label].append(generator)

        # interleave path generators
        label_generators = {k: list(interleave_generators(v) if len(v) > 1 else v[0]) for k, v in
                            label_paths.items()}

        # print(list(label_generators['whatsapp']))
        chunks = {k: self._count_chunks(v) for k, v in label_generators.items()}
        print(chunks)
        max_chunks = max(chunks.values())
        max_data_augmentation = {k: max_chunks / v for k, v in chunks.items()}
        aug_strides = {}
        for label, aug in max_data_augmentation.items():
            axis_aug = math.sqrt(aug)
            strides = [max(1, math.ceil(s / axis_aug)) for s in self.block_tile_size]
            aug_strides[label] = strides
        print(max_data_augmentation)
        print(aug_strides)

        engine = NoiseprintEngineV2()

        return {k: self._generate_examples(v, engine, aug_strides[k]) for k, v in label_generators.items()}

    def _count_chunks(self, generator):
        total = 0
        for path in generator:
            width, height = imagesize.get(path)
            total += (width // self.tile_size[0]) * (height // self.tile_size[1])
        return total

    def _generate_examples(self, dataset, engine, block_strides) -> split_builder_lib.SplitGenerator:
        cached_requests = defaultdict(list)
        strides = [s * 8 for s in block_strides]

        def process(path, image, quality):
            return self._process_image(path, image, quality, engine, strides, block_strides)

        for path in dataset:
            image, quality = open_image_quality(path)

            quality_request = cached_requests[quality]
            quality_request.append((path, image))

            if len(quality_request) < 32:
                continue

            cached_requests[quality] = []
            for request in quality_request:
                yield process(request[0], request[1], quality)

        for quality, quality_request in cached_requests.items():
            for request in quality_request:
                yield process(request[0], request[1], quality)

    def _process_image(self, path: str, image: np.ndarray, quality: int, engine: NoiseprintEngineV2, strides,
                       block_strides):
        data_path = "%s.data.npz" % path
        if not os.path.exists(data_path):
            block_dct = compute_block_dct(image).astype(np.float16)
            noiseprint = compute_noiseprint(image, quality, engine).astype(np.float16)
            np.savez(data_path, dct=block_dct, noiseprint=noiseprint)

        return path, {
            'path': path,
            'shape': image.shape[:2],
        }


def _process_entry(entry):
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


class UcidPublic(SocialImages):
    url = "http://lci.micc.unifi.it/labd/datasets/public.zip"
    rel_input_dirs = (
        ("facebook\\**\\*.jpg", 'facebook'),
        ("flickr\\**\\*.jpg", 'flickr'),
        ("twitter\\**\\*.jpg", 'twitter')
    )


class UcidSocial(SocialImages):
    url = "http://lci.micc.unifi.it/labd/datasets/ucid.zip"
    rel_input_dirs = (
        ("facebook\\**\\*.jpg", 'facebook'),
        ("flickr\\**\\*.jpg", 'flickr'),
        ("twitter\\**\\*.jpg", 'twitter')
    )


class IpLabThree(SocialImages):
    url = "https://iplab.dmi.unict.it/DigitalForensics/social_image_forensics/dataset2016.zip"
    rel_input_dirs = (
        ("**\\*facebook*\\**\\*.jpg", 'facebook'),
        ("**\\*flickr*\\**\\*.jpg", 'flickr'),
        ("**\\*twitter*\\**\\*.jpg", 'twitter')
    )

    def _on_after_download(self, dl_dir):
        to_delete = os.path.join(dl_dir, 'browser_dataset', 'FlickrDownload_old')
        if os.path.exists(to_delete):
            shutil.rmtree(to_delete)


class IpLabSeven(IpLabThree):
    rel_input_dirs = (
        ("**\\*facebook*\\**\\*.jpg", 'facebook'),
        ("**\\*flickr*\\**\\*.jpg", 'flickr'),
        ("**\\*twitter*\\**\\*.jpg", 'twitter'),
        ("**\\*imgur*\\**\\*.jpg", 'imgur'),
        ("**\\*instagram*\\**\\*.jpg", 'instagram'),
        ("**\\*telegram*\\**\\*.jpg", 'telegram'),
        ("**\\*tinypic*\\**\\*.jpg", 'tinypic'),
        ("**\\*tumblr*\\**\\*.jpg", 'tumblr'),
        ("**\\*wapp*\\**\\*.jpg", 'whatsapp'),
        ("**\\*whatsapp*\\**\\*.jpg", 'whatsapp'),
    )
