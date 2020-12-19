from tensorflow.python.keras.layers import Conv2D, BatchNormalization, Activation
from tensorflow.python.training.tracking.data_structures import NoDependency

import socialdetector.tf_options_setter
import os
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.models import Model

from socialdetector.utility import log, jpeg_qtableinv, imread2f_pil

tf1 = tf.compat.v1


def _si(value):
    v = value.value()
    return lambda shape, dtype: v


class NpConv2D(tf.keras.layers.Layer):

    def __init__(self, vl, filter_size, out_filters, stride, padding, scope_name='conv', **kwargs):
        self.vl = vl
        self._out_filters = out_filters
        self._stride = stride
        self._filter_size = filter_size
        self._padding = padding
        self.scope_name = scope_name
        self.kernel = None
        super().__init__(**kwargs)

    def build(self, input_shape):
        with tf1.variable_scope(self.scope_name):
            in_filters = input_shape[-1]
            n = self._filter_size * self._filter_size * np.maximum(in_filters, self._out_filters)
            self.kernel = tf1.get_variable(
                'weights', [self._filter_size, self._filter_size, in_filters, self._out_filters],
                tf.float32, initializer=tf1.random_normal_initializer(
                    stddev=np.sqrt(2.0 / n), dtype=tf.float32))
            self.vl.append(self.kernel)

    @tf.function
    def call(self, inputs, training=None):
        return tf.nn.conv2d(inputs, self.kernel, [1, self._stride, self._stride, 1], padding=self._padding)

    def to_keras(self):
        return Conv2D(self._out_filters, self._filter_size, (self._stride, self._stride), padding=self._padding,
                      kernel_initializer=_si(self.kernel), use_bias=False, bias_initializer="zeros")


class NpBatchNorm(tf.keras.layers.Layer):

    def __init__(self, vl, scope_name='bn', **kwargs):
        self.vl = vl
        self.moving_mean = None
        self.moving_variance = None
        self.gamma = None
        self.scope_name = scope_name
        self._bnorm_init_var = 1e-4
        self._bnorm_init_gamma = np.sqrt(2.0 / (9.0 * 64.0))
        self._bnorm_epsilon = 1e-5
        super().__init__(**kwargs)

    def build(self, input_shape):
        with tf1.variable_scope(self.scope_name):
            params_shape = input_shape[-1]

            self.moving_mean = tf1.get_variable(
                'moving_mean', params_shape, tf.float32,
                initializer=tf1.constant_initializer(0.0, dtype=tf.float32),
                trainable=False)
            self.moving_variance = tf1.get_variable(
                'moving_variance', params_shape, tf.float32,
                initializer=tf1.constant_initializer(self._bnorm_init_var, dtype=tf.float32),
                trainable=False)

            self.gamma = tf1.get_variable(
                'gamma', params_shape, tf.float32,
                initializer=tf1.random_normal_initializer(stddev=self._bnorm_init_gamma, dtype=tf.float32))

            self.vl.extend((self.moving_mean, self.moving_variance, self.gamma))

    @tf.function
    def call(self, inputs, training=None):
        return tf.nn.batch_normalization(
            inputs, self.moving_mean, self.moving_variance, None, self.gamma, self._bnorm_epsilon)

    def to_keras(self):
        return BatchNormalization(moving_mean_initializer=_si(self.moving_mean),
                                  moving_variance_initializer=_si(self.moving_variance),
                                  gamma_initializer=_si(self.gamma), epsilon=self._bnorm_epsilon)


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


class NpBias(tf.keras.layers.Layer):

    def __init__(self, vl, scope_name='bias', **kwargs):
        self.vl = vl
        self.beta = None
        self.scope_name = scope_name
        super().__init__(**kwargs)

    def build(self, input_shape):
        with tf1.variable_scope(self.scope_name):
            params_shape = input_shape[-1]
            self.b = tf1.get_variable(
                'beta', params_shape, tf.float32,
                initializer=tf1.constant_initializer(0.0, dtype=tf.float32))
            self.vl.append(self.b)

    @tf.function
    def call(self, inputs, training=None):
        return inputs + self.b

    def to_keras(self):
        return BiasLayer(self.b.value())


class NpActivation(tf.keras.layers.Layer):

    def __init__(self, act_fun, scope_name='active', **kwargs):
        self.act_fun = act_fun
        self.scope_name = scope_name
        super().__init__(**kwargs)

    @tf.function
    def call(self, inputs, training=None):
        return self.act_fun(inputs)

    def to_keras(self):
        return Activation(self.act_fun)


class FullConvNet(object):
    """FullConvNet model."""

    def __init__(self, num_levels=17, padding='SAME'):
        """FullConvNet constructor."""

        self._num_levels = num_levels
        self._actfun = [tf.nn.relu, ] * (self._num_levels - 1) + [tf.identity, ]
        self._f_num = [64, ] * (self._num_levels - 1) + [1, ]
        self._bnorm = [False, ] + [True, ] * (self._num_levels - 2) + [False, ]

        self.variables_list = []
        self.k_input = tf.keras.layers.Input([None, None, 1])

        model = self.k_input
        vl = NoDependency(self.variables_list)

        seq = []

        def add_layer(lay):
            seq.append(lay)
            return lay

        for i in range(self._num_levels):
            with tf1.variable_scope('level_%d' % i):
                model = add_layer(NpConv2D(vl, 3, self._f_num[i], 1, padding=padding))(model)
                if self._bnorm[i]:
                    model = add_layer(NpBatchNorm(vl))(model)
                model = add_layer(NpBias(vl))(model)
                model = add_layer(NpActivation(self._actfun[i]))(model)

        self.model = Model(self.k_input, model)
        self.model.summary()
        self.seq = seq

    def run(self, x):
        return self.model.predict(x)

    def to_keras(self):
        i = tf.keras.layers.Input([None, None, 1])
        model = i
        for l in self.seq:
            model = l.to_keras()(model)
        return Model(i, model)


class FullConvNetV2(object):
    """FullConvNet model."""

    def __init__(self, num_levels=17, padding='SAME'):
        """FullConvNet constructor."""

        self._actfun = [tf.nn.relu, ] * (num_levels - 1) + [tf.identity, ]
        self._f_num = [64, ] * (num_levels - 1) + [1, ]
        self._bnorm = [False, ] + [True, ] * (num_levels - 2) + [False, ]

        inp = tf.keras.layers.Input([None, None, 1])
        model = inp

        for i in range(num_levels):
            model = Conv2D(self._f_num[i], 3, padding=padding, use_bias=False)(model)
            if self._bnorm[i]:
                model = BatchNormalization(epsilon=1e-5)(model)
            model = BiasLayer()(model)
            model = Activation(self._actfun[i])(model)

        self.model = Model(inp, model)
        self.model.summary()





class NoiseprintEngine:
    net = FullConvNet()
    saver = tf1.train.Saver(net.variables_list)
    checkpoint_template = os.path.join(os.path.dirname(__file__), './noiseprint/net_jpg%d/model')
    save_p = os.path.join(os.path.dirname(__file__), './noiseprint_V2/net_jpg%d/')
    configSess = tf1.ConfigProto()
    configSess.gpu_options.allow_growth = True
    slide = 1024  # 3072
    largeLimit = 1050000  # 9437184
    overlap = 34

    def __init__(self, quality=101):
        self.quality = quality
        self.model = self.net
        self.loaded_quality = None
        self.session = tf1.Session()
        self.load_session(quality)

    def load_session(self, quality):
        log("Setting quality to %d " % quality)
        if quality == self.loaded_quality:
            return
        if quality < 51:
            quality = 51
        elif quality > 100:
            quality = 101
        log("Reloading checkpoint %d " % quality)
        checkpoint = self.checkpoint_template % quality
        s_c = self.save_p % quality

        self.saver.restore(self.session, checkpoint)

        k_model = self.model.to_keras()

        k_model.save_weights(s_c)
        # k_model.load_weights(s_c)

        self.k_model = k_model
        # exit(1)
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
                resB = self.model.run(clip[np.newaxis, :, :, np.newaxis])
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
        res = self.model.run(img[np.newaxis, :, :, np.newaxis])
        return np.squeeze(res)

    def predict(self, img):
        if img.shape[0] * img.shape[1] > self.largeLimit:
            return self._predict_large(img)
        else:
            return self._predict_small(img)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def gen_noiseprint(image, quality=None):
    if isinstance(image, str):
        if quality is None:
            quality = jpeg_qtableinv(image)
        image = imread2f_pil(image, channel=1)[0]
    else:
        if quality is None:
            quality = 101
    with NoiseprintEngine(quality) as engine:
        return engine.predict(image), engine.k_model.predict(image[tf.newaxis, ..., tf.newaxis])


def load_all():
    engine = NoiseprintEngine()
    for i in range(50, 102):
        engine.load_session(i)

load_all()


def normalize_noiseprint(noiseprint, margin=34):
    v_min = np.min(noiseprint[margin:-margin, margin:-margin])
    v_max = np.max(noiseprint[margin:-margin, margin:-margin])
    return ((noiseprint - v_min) / (v_max - v_min)).clip(0, 1)
