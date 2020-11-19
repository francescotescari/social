from scipy.fftpack import dct, idct
import numpy as np


def _build_fixed_array(value, size):
    return [value for i in range(size)]


coefficient_order = np.array(
    [1, 9, 2, 3, 10, 17, 25, 18, 11, 4, 5, 12, 19, 26, 33, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 57, 50, 43,
     36, 29, 22, 15, 8, 16, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 24, 32, 39, 46, 53, 60, 61, 54, 47, 40, 48, 55,
     62, 63, 56, 64]) - 1
standard_block_size = 8
quantization_limit = np.array(
    _build_fixed_array(20, 10) + _build_fixed_array(30, 5) + _build_fixed_array(20, 10) + _build_fixed_array(40,
                                                                                                             6) + _build_fixed_array(
        64, 7) + _build_fixed_array(80, 8) + _build_fixed_array(100, 28))


def dct2(a, norm='ortho', type=2):
    return dct(dct(a, axis=-1, norm=norm, type=type), axis=-2, norm=norm, type=type)


def idct2(a, norm='ortho', type=2):
    return idct(idct(a, axis=-1, norm=norm, type=type), axis=-2, norm=norm, type=type)


def round_direction(d, size=standard_block_size, add_padding=True):
    return ((np.ceil if add_padding else np.floor)(d / size) * size).astype(np.int)


def assure_dimension(img, dims, pad_mode='reflect'):
    dims = np.array(dims)
    size = np.array(img.shape[:len(dims)])
    diff = (dims - size).astype(np.int)
    if np.any(diff < 0):
        slices = [slice(0, d) if d < 0 else slice(None) for d in diff]
        img = img[tuple(slices)]
    if np.any(diff > 0):
        pad = np.array([[0, x] for x in np.clip(diff, 0, None)])
        img = np.pad(img, pad, mode=pad_mode)
    return img


def assure_blockable(img, block_size=standard_block_size, add_padding=True):
    wanted_shape = tuple([round_direction(img.shape[i], add_padding=add_padding, size=block_size) for i in (0, 1)])
    return assure_dimension(img, wanted_shape)


def to_block_array(img, block_size=standard_block_size):
    if img.shape[0] % block_size != 0 or img.shape[1] % block_size != 0:
        raise ValueError("Invalid img shape and block size: {} - {}".format(img.shape, block_size))
    return np.array(
        [img[x:x + 8, y:y + 8] for x in range(0, img.shape[0], block_size) for y in range(0, img.shape[1], block_size)])

def to_block_matrix(img, block_size=standard_block_size):
    if img.shape[0] % block_size != 0 or img.shape[1] % block_size != 0:
        raise ValueError("Invalid img shape and block size: {} - {}".format(img.shape, block_size))
    return np.array([[img[x:x + 8, y:y + 8] for x in range(0, img.shape[0], block_size)] for y in range(0, img.shape[1], block_size)])


def blockwise_dct_matrix(img, add_padding=True, block_size=standard_block_size):
    blocks = to_block_matrix(assure_blockable(img, add_padding=add_padding, block_size=block_size),
                            block_size=block_size)
    dct_blocks = dct2(blocks, type=2)
    return dct_blocks

def blockwise_dct_array(img, add_padding=True, block_size=standard_block_size):
    blocks = to_block_array(assure_blockable(img, add_padding=add_padding, block_size=block_size),
                            block_size=block_size)
    dct_blocks = dct2(blocks, type=2)
    return dct_blocks


def to_coefficient_map(blocks, block_size=standard_block_size):
    size = blocks.shape[0]
    return np.reshape(blocks, (size, block_size * block_size)).swapaxes(0, -1)


def to_blocks_array(coefficients_values, block_size=standard_block_size):
    size = coefficients_values.shape[1]
    return coefficients_values.swapaxes(0, -1).reshape((size, block_size, block_size))
