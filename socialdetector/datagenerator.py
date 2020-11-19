import os
import numpy as np
import PIL.Image as Image

from socialdetector.dct_utils import blockwise_dct_matrix
from socialdetector.noiseprint import gen_noiseprint, NoiseprintEngine
from socialdetector.utility import log, imread2f_pil, jpeg_qtableinv, FileWalker, imread_mode


class PathTupleGenerator:

    def __init__(self, origin_files: FileWalker, destination_dirs: dict, origin_file_key='origin_file', origin_dir=None, ):
        self.walker = origin_files
        self.destination_dirs = destination_dirs
        self.origin_key = origin_file_key
        self.generated = None
        self.origin_dir = origin_files.origin_dir if origin_dir is None else origin_dir

    def generate(self):
        if self.generated is None:
            files = []
            for filename in self.walker:
                dir_path, name = os.path.split(os.path.abspath(filename))
                relative_dir_path = os.path.relpath(dir_path, self.origin_dir)
                entry = {self.origin_key: filename}
                for key, path in self.destination_dirs.items():
                    output_dir = os.path.join(path, relative_dir_path)
                    output_filename = os.path.join(output_dir, name)
                    entry[key] = output_filename
                files.append(entry)
            self.generated = files
        return self.generated

    def __iter__(self):
        return iter(self.generate())

    def __len__(self):
        return len(self.generate())

    def __getitem__(self, item):
        return self.generate()[item]


class NoiseprintDataGenerator:

    def __init__(self, file_tuple_generator: PathTupleGenerator, source_key='noiseprint_path', dest_key='noiseprint'):
        self.path_generator = file_tuple_generator
        self.source_key = source_key
        self.dest_key = dest_key

    def __getitem__(self, item):
        data = {**self.path_generator[item]}
        noiseprint_path = data[self.source_key]
        noiseprint_path += ".npz"
        with np.load(noiseprint_path) as noiseprint_file:
            noiseprint_data = noiseprint_file[noiseprint_file.files[0]]
        data[self.dest_key] = noiseprint_data
        return data

    def __len__(self):
        return len(self.path_generator)

    def generate_and_save(self, origin_file_key='origin_file'):
        with NoiseprintEngine() as engine:
            for entry in self.path_generator:
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


class TransformingDataGenerator:

    def __init__(self, parent_generator, source_key, destination_key):
        self.parent = parent_generator
        self.source_key = source_key
        self.destination_key = destination_key

    def __len__(self):
        return len(self.parent)

    def __getitem__(self, item):
        data = {**self.parent[item]}
        print(str(self.__class__), len(data), type(data))
        if self.source_key is None:
            self.transform(data)
        else:
            data[self.destination_key] = self.transform(data[self.source_key])

        return data

    def transform(self, data):
        raise NotImplementedError()


class ImageLoadingGenerator(TransformingDataGenerator):
    MODES = ["YCbCr", "RGB", "L"]

    def __init__(self, parent_generator, source_key=None, destination_key=None, mode="RGB", dtype=np.float32):
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
        return blockwise_dct_matrix((data[..., 0]).astype(np.float) - 128)


class MultipleTransformingDataGenerator:

    def __init__(self, parent_generator, multiplier=1, bring_properties=tuple()):
        self.parent = parent_generator
        self.multiplier = multiplier
        self.cached_items = []
        self.cursors = 0
        self.bring_properties = bring_properties

    def __len__(self):
        return len(self.parent) * self.multiplier

    def __getitem__(self, item):
        if len(self.cached_items) > 0:
            return self.cached_items.pop(0)
        data = {**self.parent[self.cursors]}
        self.cursors += 1
        self.cached_items = self.transform(data)
        for prop in self.bring_properties:
            for item in self.cached_items:
                item[prop] = data[prop]
        return self[item]

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
                for y in range(0, img_height, slide_step_y):
                    slice = img[x:x + ww, y:y + wh, ...]
                    slices.append(slice)
            return slices

        splices = {}
        max_splices = 0
        for entry_key, mul in self.entries.items():
            s = split(data[entry_key], mul)
            splices[entry_key] = s
            max_splices = max(max_splices, len(s))
        return [{entry_key+self.destination_suffix: splices[entry_key][i] for entry_key in splices} for i in range(max_splices)]
