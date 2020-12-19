from tensorflow.python.keras.layers import Conv2D, BatchNormalization, Activation
from tensorflow.python.training.tracking.data_structures import NoDependency

import socialdetector.tf_options_setter
import os
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.models import Model

from socialdetector.utility import log, jpeg_qtableinv, imread2f_pil


class BiasLayer(tf.keras.layers.Layer):

    def __init__(self, initial_value=None, **kwargs):
        super(BiasLayer, self).__init__(**kwargs)
        self.val = initial_value

    def build(self, input_shape):
        if self.val is None:
            self.bias = self.add_weight('bias', shape=input_shape[-1], initializer="zeros")
        else:
            self.bias = self.add_weight('bias', shape=input_shape[-1],
                                        initializer=lambda shape, dtype: tf.cast(self.val, dtype=dtype))

    @tf.function
    def call(self, inputs, training=None):
        return inputs + self.bias


def _FullConvNetV2(num_levels=17, padding='SAME'):
    """FullConvNet model."""

    activation_fun = [tf.nn.relu, ] * (num_levels - 1) + [tf.identity, ]
    filters_num = [64, ] * (num_levels - 1) + [1, ]
    batch_norm = [False, ] + [True, ] * (num_levels - 2) + [False, ]

    inp = tf.keras.layers.Input([None, None, 1])
    model = inp

    for i in range(num_levels):
        model = Conv2D(filters_num[i], 3, padding=padding, use_bias=False)(model)
        if batch_norm[i]:
            model = BatchNormalization(epsilon=1e-5)(model)
        model = BiasLayer()(model)
        model = Activation(activation_fun[i])(model)

    return Model(inp, model)


class NoiseprintEngineV2:
    model = _FullConvNetV2()
    save_path = os.path.join(os.path.dirname(__file__), './noiseprint_V2/net_jpg%d/')
    configSess = tf.compat.v1.ConfigProto()
    configSess.gpu_options.allow_growth = True
    slide = 1024  # 3072
    largeLimit = 1050000  # 9437184
    overlap = 34

    def __init__(self, quality=None):
        self.quality = quality
        self.loaded_quality = None
        if quality is not None:
            self.load_session(quality)

    def load_session(self, quality):
        log("Setting quality to %d " % quality)
        quality = min(max(quality, 51), 101)
        if quality == self.loaded_quality:
            return
        log("Reloading checkpoint %d " % quality)
        checkpoint = self.save_path % quality
        self.model.load_weights(checkpoint)
        self.loaded_quality = quality

    def _predict_large(self, img):
        res = np.zeros((img.shape[0], img.shape[1]), np.float32)
        for index0 in range(0, img.shape[0], self.slide):
            index0start = index0 - self.overlap
            index0end = index0 + self.slide + self.overlap

            for index1 in range(0, img.shape[1], self.slide):
                index1start = index1 - self.overlap
                index1end = index1 + self.slide + self.overlap
                clip = img[max(index0start, 0): min(index0end, img.shape[0]), \
                       max(index1start, 0): min(index1end, img.shape[1])]
                resB = self._predict_small(clip[np.newaxis, :, :, np.newaxis])
                resB = np.squeeze(resB)

                if index0 > 0:
                    resB = resB[self.overlap:, :]
                if index1 > 0:
                    resB = resB[:, self.overlap:]
                resB = resB[:min(self.slide, resB.shape[0]), :min(self.slide, resB.shape[1])]

                res[index0: min(index0 + self.slide, res.shape[0]), \
                index1: min(index1 + self.slide, res.shape[1])] = resB
        return res

    def _predict_small(self, img):
        return self.model.predict(img)

    def predict(self, img):
        if img.shape[0] * img.shape[1] > self.largeLimit:
            return self._predict_large(img)
        else:
            return tf.squeeze(self._predict_small(img[np.newaxis, :, :, np.newaxis]))

    @tf.function
    def predict_graphed(self, img):
        return self.predict(img)


def gen_noiseprint(image, quality=None):
    if isinstance(image, str):
        if quality is None:
            quality = jpeg_qtableinv(image)
        image = imread2f_pil(image, channel=1)[0]
    else:
        if quality is None:
            quality = 101
    return NoiseprintEngineV2(quality).predict(image)


def normalize_noiseprint(noiseprint, margin=34):
    v_min = np.min(noiseprint[margin:-margin, margin:-margin])
    v_max = np.max(noiseprint[margin:-margin, margin:-margin])
    return ((noiseprint - v_min) / (v_max - v_min)).clip(0, 1)
