import glob
import os
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.JpegImagePlugin import convert_dict_qtables


def log(*args, **kwargs):
    print("[SD]", *args, **kwargs)


def jpeg_quality_of(image, tnum=0, force_baseline=None):
    assert tnum == 0 or tnum == 1, 'Table number must be 0 or 1'

    if force_baseline is None:
        th_high = 32767
    elif force_baseline == 0:
        th_high = 32767
    else:
        th_high = 255

    h = np.asarray(convert_dict_qtables(image.quantization)[tnum]).reshape((8, 8))

    if tnum == 0:
        # This is table 0 (the luminance table):
        t = np.array(
            [[16, 11, 10, 16, 24, 40, 51, 61],
             [12, 12, 14, 19, 26, 58, 60, 55],
             [14, 13, 16, 24, 40, 57, 69, 56],
             [14, 17, 22, 29, 51, 87, 80, 62],
             [18, 22, 37, 56, 68, 109, 103, 77],
             [24, 35, 55, 64, 81, 104, 113, 92],
             [49, 64, 78, 87, 103, 121, 120, 101],
             [72, 92, 95, 98, 112, 100, 103, 99]])

    elif tnum == 1:
        # This is table 1 (the chrominance table):
        t = np.array(
            [[17, 18, 24, 47, 99, 99, 99, 99],
             [18, 21, 26, 66, 99, 99, 99, 99],
             [24, 26, 56, 99, 99, 99, 99, 99],
             [47, 66, 99, 99, 99, 99, 99, 99],
             [99, 99, 99, 99, 99, 99, 99, 99],
             [99, 99, 99, 99, 99, 99, 99, 99],
             [99, 99, 99, 99, 99, 99, 99, 99],
             [99, 99, 99, 99, 99, 99, 99, 99]])

    else:
        raise ValueError(tnum, 'Table number must be 0 or 1')

    h_down = np.divide((2 * h - 1), (2 * t))
    h_up = np.divide((2 * h + 1), (2 * t))
    if np.all(h == 1): return 100
    x_down = (h_down[h > 1]).max()
    x_up = (h_up[h < th_high]).min() if (h < th_high).any() else None
    if x_up is None:
        s = 1
    elif x_down > 1 and x_up > 1:
        s = np.ceil(50 / x_up)
    elif x_up < 1:
        s = np.ceil(50 * (2 - x_up))
    else:
        s = 50
    return s


def jpeg_qtableinv(stream, tnum=0, force_baseline=None):
    return jpeg_quality_of(Image.open(stream), tnum=tnum, force_baseline=force_baseline)


def image_to_channels(img, channel=1, dtype=np.float32):
    mode = img.mode

    if channel == 3:
        img = img.convert('RGB')
        img = np.asarray(img).astype(dtype) / 256.0
    elif channel == 1:
        if img.mode == 'L':
            img = np.asarray(img).astype(dtype) / 256.0
        else:
            img = img.convert('RGB')
            img = np.asarray(img).astype(dtype)
            img = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]) / 256.0
    else:
        img = np.asarray(img).astype(dtype) / 256.0
    return img, mode


def imread2f_pil(stream, channel=1, dtype=np.float32):
    return image_to_channels(Image.open(stream), channel=channel, dtype=dtype)


def imread_mode(filename, mode="RGB", dtype=np.float32):
    image = np.asarray(Image.open(filename).convert(mode))
    orig_type = image.dtype
    if np.issubdtype(orig_type, np.integer):
        return image if np.issubdtype(dtype, np.integer) else image.astype(np.float32) / 256
    else:
        return (image * 256).astype(np.int32) if np.issubdtype(dtype, np.integer) else image


class _TreeIterator:

    def __init__(self, *iterators_gen):
        self.depth = len(iterators_gen)
        self.loaded_iterators = [None for _ in range(self.depth)]
        self.last_i = self.depth - 1
        self.gen = iterators_gen
        self.loaded_iterators[0] = iterators_gen[0](None)

    def __iter__(self):
        return _TreeIterator(*self.gen)

    def _next_i(self, i):
        it = self.loaded_iterators[i]
        while True:
            if it is None:
                it = self.gen[i](self._next_i(i - 1))
                self.loaded_iterators[i] = it
            try:
                return next(it)
            except StopIteration:
                if i == 0:
                    raise
                else:
                    it = None

    def __next__(self):
        return self._next_i(self.last_i)


def interleave_generators(gens):
    gens = list(gens)
    while True:
        for gen in gens:
            try:
                yield next(gen)
            except StopIteration:
                gens.remove(gen)
                if len(gens) == 0:
                    return


def count_iterator(iterator):
    i = 0
    try:
        while True:
            next(iterator)
            i += 1
    except StopIteration:
        pass
    return i


def path_glob_iterator(path, pattern, recursive=True):
    p = Path(path)
    it = p.rglob(pattern) if recursive else p.glob(pattern)
    return map(lambda x: str(x), it)


def glob_iterator(pattern, recursive=True):
    return glob.iglob(pattern, recursive=recursive)


class FileWalker:

    def __init__(self, origin_dir, filter_function=None):
        self.origin_dir = origin_dir
        self.filter_function = filter_function
        self.file_iterators = None

    def __next__(self):
        while True:
            if self.file_iterators is None:
                file_iterator = iter(os.walk(self.origin_dir, topdown=False))
                root, dirs, files = next(file_iterator)
                self.file_iterators = [file_iterator, (root, iter(files))]
            while True:
                root, fi = self.file_iterators[1]
                try:
                    name = next(fi)
                except StopIteration:
                    root, _, files = next(self.file_iterators[0])
                    self.file_iterators[1] = (root, iter(files))
                else:
                    break
            path = os.path.join(root, name)
            if self.filter_function is None or self.filter_function(path):
                return path

    def __iter__(self):
        return FileWalker(self.origin_dir, self.filter_function)


def isiterable(o):
    try:
        iter(o)
    except:
        return False
    else:
        return True


def is_listing(o):
    return isinstance(o, list) or isinstance(o, tuple)


def split_filters(generator, *filters):
    filters = list(filters)
    res = [[] for _ in filters] + [[]]
    tot = len(res)-1
    for entry in generator:
        i = 0
        for fil in filters:
            if fil(entry):
                res[i].append(entry)
                break
            i += 1
        if i == tot:
            res[-1].append(entry)
    return res
