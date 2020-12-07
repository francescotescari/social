import os
import random

import numpy as np
import PIL.Image as Image
from tensorflow.python.data import Dataset

from socialdetector.dct_utils import blockwise_dct_matrix, coefficient_order
from socialdetector.noiseprint import gen_noiseprint, NoiseprintEngine
from socialdetector.utility import log, imread2f_pil, jpeg_qtableinv, FileWalker, imread_mode, isiterable
import matplotlib.pyplot as plt


class DataFinished(Exception):
    pass


class DataGenerator:

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        return GetterDataGen(self, item)


class GetterDataGen(DataGenerator):

    def __init__(self, parent_generator, keys):
        self.parent = parent_generator
        self.keys = keys

    def __next__(self):
        data = next(self.parent)
        if isinstance(self.keys, list) or isinstance(self.keys, tuple):
            return [data[k] for k in self.keys]
        else:
            return data[self.keys]


class TransformingDataGenerator(DataGenerator):

    def __init__(self, parent_generator, source_key, destination_key):
        self.parent = parent_generator
        self.source_key = source_key
        self.destination_key = destination_key
        if source_key is None:
            self.transform_fn = self.transform_whole
        elif isinstance(source_key, tuple) or isinstance(source_key, list):
            self.transform_fn = self.transform_multi_key
        else:
            self.transform_fn = self.transform_single_key

    def __len__(self):
        return len(self.parent)

    def __next__(self):
        return self.transform_fn({**next(self.parent)})

    def transform(self, data):
        raise NotImplementedError()

    def transform_whole(self, data):
        self.transform(data)
        return data

    def transform_single_key(self, data):
        data[self.destination_key] = self.transform(data[self.source_key])
        return data

    def transform_multi_key(self, data):
        data[self.destination_key] = self.transform([data[k] for k in self.source_key])
        return data


class PathTupleGenerator(DataGenerator):

    def __next__(self):
        return self._process_entry(next(self.walker))

    def __init__(self, origin_files: FileWalker, destination_dirs: dict, origin_file_key='origin_file',
                 origin_dir=None, ):
        self.walker = origin_files
        self.destination_dirs = destination_dirs
        self.origin_key = origin_file_key
        self.generated = None
        self.origin_dir = origin_files.origin_dir if origin_dir is None else origin_dir
        self.file_iterator = iter(self.walker)

    def _process_entry(self, filename):
        dir_path, name = os.path.split(os.path.abspath(filename))
        relative_dir_path = os.path.relpath(dir_path, self.origin_dir)
        entry = {self.origin_key: filename}
        for key, path in self.destination_dirs.items():
            output_dir = os.path.join(path, relative_dir_path)
            output_filename = os.path.join(output_dir, name)
            entry[key] = output_filename
        # print("D", entry)
        return entry


class NoiseprintDataGenerator(TransformingDataGenerator):

    def transform(self, data):
        noiseprint_path = data + ".npz"
        with np.load(noiseprint_path) as noiseprint_file:
            return noiseprint_file[noiseprint_file.files[0]]

    def generate_and_save(self, origin_file_key='origin_file'):
        with NoiseprintEngine() as engine:
            for entry in self.parent:
                filename, output_filename = entry[origin_file_key], entry[self.source_key]
                output_filename += ".npz"
                if os.path.isfile(output_filename):
                    log("Skipping present file: %s" % output_filename)
                    continue
                output_dir = os.path.split(os.path.abspath(filename))[1]
                try:
                    os.makedirs(output_dir)
                except FileExistsError:
                    pass

                quality = jpeg_qtableinv(filename)
                image = imread2f_pil(filename, channel=1)[0]
                engine.load_session(quality)
                noiseprint = engine.predict(image)
                np.savez(output_filename, noiseprint.astype(np.float16))
                log("Saved noiseprint: %s" % output_filename)


class ImageLoadingGenerator(TransformingDataGenerator):
    MODES = ["YCbCr", "RGB", "L"]

    def __init__(self, parent_generator, source_key=None, destination_key=None, mode="RGB", dtype=np.uint8):
        if mode not in self.MODES:
            raise ValueError("Unknown mode: %s" % mode)
        self.mode = mode
        self.dtype = dtype
        super().__init__(parent_generator, source_key=source_key, destination_key=destination_key)

    def transform(self, data):
        return imread_mode(data, mode=self.mode, dtype=self.dtype)


class JpegQuantizationTableGenerator(TransformingDataGenerator):

    def transform(self, data):
        try:
            q = Image.open(data).quantization
        except AttributeError:
            q = None
        return q


class JpegBlockDctGenerator(TransformingDataGenerator):

    def transform(self, data):
        return blockwise_dct_matrix((data[..., 0]).astype(np.int16) - 128)


class MultipleTransformingDataGenerator(DataGenerator):

    def __init__(self, parent_generator, multiplier=1, bring_properties=tuple()):
        self.parent = parent_generator
        self.multiplier = multiplier
        self.cached_items = []
        self.bring_properties = bring_properties

    def __len__(self):
        return len(self.parent) * self.multiplier

    def __next__(self):
        if len(self.cached_items) > 0:
            return self.cached_items.pop(0)
        data = {**next(self.parent)}
        self.cached_items = self.transform(data)
        for prop in self.bring_properties:
            for item in self.cached_items:
                item[prop] = data[prop]
        return next(self)

    def transform(self, data) -> list:
        raise NotImplementedError


class ImageSplitterDataGenerator(MultipleTransformingDataGenerator):

    def __init__(self, parent_generator, window_block_shape, multiply_entries={}, destination_suffix='splitted',
                 block_overlap=(1, 1), **kwargs):
        super().__init__(parent_generator=parent_generator, **kwargs)
        self.windows_shape = window_block_shape
        self.entries = multiply_entries
        self.destination_suffix = destination_suffix
        self.window_overlap = block_overlap

    def transform(self, data) -> list:
        window_width = self.windows_shape[0]
        window_height = self.windows_shape[1]

        def split(img, multiplier=1):
            ww = window_width * multiplier
            wh = window_height * multiplier
            slide_step_x = (window_width - self.window_overlap[0]) * multiplier
            slide_step_y = (window_height - self.window_overlap[1]) * multiplier
            img_width, img_height = img.shape[0], img.shape[1]
            slices = []
            for x in range(0, img_width, slide_step_x):
                rx = min(x, img_width - ww)
                for y in range(0, img_height, slide_step_y):
                    ry = min(y, img_height - wh)
                    slice = img[rx:rx + ww, ry:ry + wh, ...]
                    slices.append(slice)
            return slices

        splices = {}
        max_splices = 0
        for entry_key, mul in self.entries.items():
            s = split(data[entry_key], mul)
            splices[entry_key] = s
            max_splices = max(max_splices, len(s))
        return [{entry_key + self.destination_suffix: splices[entry_key][i] for entry_key in splices} for i in
                range(max_splices)]


class StaticDataGenerator(DataGenerator):

    def __next__(self):
        if self.done:
            raise StopIteration()
        if self.once:
            self.done = True
        return self.data

    def __init__(self, data, once=False):
        self.data = data
        self.once = once
        self.done = False


num_considered_coefficients = 10


class JpegDctFirstQuantizers(TransformingDataGenerator):

    def transform(self, data):
        return np.array(data[0][1:num_considered_coefficients])


class JpegDctFirstCoefficient(TransformingDataGenerator):
    first_coefficients = tuple(coefficient_order[1:num_considered_coefficients])

    print(first_coefficients)

    def transform(self, data):
        shp = data.shape
        return data.reshape((shp[0] * shp[1], shp[2] * shp[3])).swapaxes(0, -1)[self.first_coefficients, ...]


class DefaultJpegPaperEncoding(TransformingDataGenerator):

    def transform(self, data):
        data = data.clip(-50, 50)
        return np.apply_along_axis(lambda x: np.histogram(x, 101, (-50.5, 50.5))[0], -1, data).reshape(909, 1).astype(
            np.float32) / data.shape[1]


class BufferDataGenerator(DataGenerator):

    def __next__(self):
        self._ensure_buffer()
        return self._next_in_buffer()

    def __init__(self, parent_generator, buffer_size):
        self.parent = parent_generator
        self.buffer_size = buffer_size
        self.buffer = []
        self.parent_empty = False

    def _ensure_buffer(self):
        if self.parent_empty:
            return
        try:
            while len(self.buffer) < self.buffer_size:
                self.buffer.append(next(self.parent))
        except StopIteration:
            self.parent_empty = True

    def _next_in_buffer(self):
        try:
            return self.buffer.pop(0)
        except IndexError:
            raise StopIteration()


class ShuffleDataGenerator(BufferDataGenerator):

    def __next__(self):
        self._ensure_buffer()
        random.shuffle(self.buffer)
        return self._next_in_buffer()


class ConcatDataGenerator(DataGenerator):

    def __init__(self, sub_generators):
        self.sub_generators = sub_generators
        self.cursor = 0
        self.current = self.sub_generators[self.cursor]

    def __next__(self):
        try:
            return next(self.current)
        except StopIteration:
            self.cursor += 1
            if self.cursor < len(self.sub_generators):
                self.current = self.sub_generators[self.cursor]
                return next(self)
            else:
                raise StopIteration()


class AlternateDataGenerator(DataGenerator):

    def __init__(self, sub_generators):
        self.sub_generators: list = [s for s in sub_generators]
        self.cursor = -1
        self.len = len(self.sub_generators)

    def __next__(self):
        self.cursor = (self.cursor + 1) % self.len
        try:
            return next(self.sub_generators[self.cursor])
        except StopIteration:
            self.sub_generators.pop(self.cursor)
            self.cursor -= 1
            self.len = len(self.sub_generators)
            if self.len == 0:
                raise StopIteration
            return next(self)


class LogDataGenerator(TransformingDataGenerator):

    def transform(self, data):
        log(data)
        return data


class AppendDataDataGenerator(TransformingDataGenerator):

    def __init__(self, parent_generator, data, destination_key=None):
        if destination_key is None and not isinstance(data, dict):
            raise ValueError("Cannot pass non-dict data without a destination")
        super().__init__(parent_generator, None, None)
        self.dd_key = destination_key
        self.data = data

    def transform(self, data):
        if self.dd_key is not None:
            data[self.dd_key] = self.data
        else:
            for k in self.data:
                data[k] = self.data[k]
        return data


class InputOutputDataGenerator(DataGenerator):

    def __next__(self):
        data = next(self.parent)
        return data[self.input_key], data[self.output_key]

    def __init__(self, parent_generator, input_key, output_key):
        self.parent = parent_generator
        self.input_key = input_key
        self.output_key = output_key


class CustomTransformDataGenerator(TransformingDataGenerator):

    def __init__(self, parent_generator, source_key, destination_key, transform):
        super().__init__(parent_generator, source_key, destination_key)
        self.tr = transform

    def transform(self, data):
        return self.tr(data)


class HomogeneousPollingDataGenerator(AlternateDataGenerator):
    pass


def path_to_cnn_generator(path_generator, source_key='origin_file', destination_key='input', overlap_patches=(1, 1),
                          jpeg_coeff_encoding=DefaultJpegPaperEncoding):
    gen = JpegQuantizationTableGenerator(path_generator, source_key=source_key, destination_key='quantization')
    gen = ImageLoadingGenerator(gen, source_key='origin_file', mode="YCbCr", destination_key='ycbcrImage')
    gen = JpegBlockDctGenerator(gen, source_key='ycbcrImage', destination_key='blockDct')
    gen = ImageSplitterDataGenerator(gen, (8, 8), {'blockDct': 1}, destination_suffix='_split',
                                     block_overlap=overlap_patches)  ### , 'noiseprint': 8
    gen = JpegDctFirstCoefficient(gen, source_key='blockDct_split', destination_key='encoded')
    gen = jpeg_coeff_encoding(gen, source_key='encoded', destination_key=destination_key)
    return gen
