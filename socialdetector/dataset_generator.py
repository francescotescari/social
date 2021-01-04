from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.models import Model

from socialdetector.dataset_utils import *
from socialdetector.dct_utils import coefficient_order


def reshape_block_dct(considered_coefficients=10):
    considered_coefficients = coefficient_order[1:1 + considered_coefficients]

    def apply(x):
        if len(x.shape) > 3:
            shape = [tf.shape(x)[k] for k in range(len(x.shape))]
            x = tf.reshape(x, (*shape[:-2], 64))
        x = tf.cast(x, tf.float32)
        return tf.gather(x, considered_coefficients, axis=-1)[tf.newaxis, ...]

    return apply


def encode_coefficients_paper(considered_coefficients=10):
    def apply(x):
        x = tf.reshape(x, (-1, considered_coefficients))
        x = tf.einsum("i...j->j...i", x)
        x = tf.clip_by_value(x, -50, 50)
        size = x.shape[-1]
        x = tf.map_fn(lambda a: tf.histogram_fixed_width(a, (-50.5, 50.5), nbins=101, dtype=tf.int32), x,
                      fn_output_signature=tf.int32)
        x = tf.reshape(x, (-1,))

        return x[..., tf.newaxis]

    return apply


@tf.function
def normalize_max_min(t, axis=None):
    t = tf.cast(t, tf.float32)
    t_max = tf.reduce_max(t, axis=axis)
    t_min = tf.reduce_min(t, axis=axis)
    if axis is not None:
        t_max = tf.expand_dims(t_max, axis=axis)
        t_min = tf.expand_dims(t_min, axis=axis)
    #tf.print("A", t_min, "B", t_max, "\n")
    diff = (t_max - t_min)
    diff += tf.where(diff == 0,1.0, 0.0)
    return 2*((t - t_min) / diff)-1

@tf.function
def normalize(t, axis=None):
    t = tf.cast(t, tf.float32)
    m = tf.reduce_mean(t, axis=axis)
    s = tf.math.reduce_std(t, axis=axis)
    if axis is not None:
        t_max = tf.expand_dims(m, axis=axis)
        t_min = tf.expand_dims(s, axis=axis)
    #tf.print("A", t_min, "B", t_max, "\n")
    s += tf.where(s == 0, 1.0, 0.0)
    return (t-m)/s

def encode_coefficients_my(considered_coefficients=10):
    def apply(x):
        x = tf.reshape(x, (-1, considered_coefficients))
        x = tf.einsum("i...j->j...i", x)
        size = x.shape[-1]
        results = []
        for i in range(1, 21):
            tmp = x / i
            diff = tmp - tf.round(tmp)
            results.append([tf.histogram_fixed_width(diff[j], (-0.5, 0.5), 11) for j in range(considered_coefficients)])

        x = tf.convert_to_tensor(results)
        x = tf.einsum("ij...->ji...", x)
        #return x
        return normalize(x)

    return apply
