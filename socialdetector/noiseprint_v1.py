import os
from time import time

import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()

from socialdetector.utility import log, jpeg_qtableinv, imread2f_pil


class FullConvNet(object):
    """FullConvNet model."""

    def __init__(self, images, bnorm_decay, falg_train, num_levels=17, padding='SAME'):
        """FullConvNet constructor."""

        self._num_levels = num_levels
        self._actfun = [tf.nn.relu, ] * (self._num_levels - 1) + [tf.identity, ]
        self._f_size = [3, ] * self._num_levels
        self._f_num = [64, ] * (self._num_levels - 1) + [1, ]
        self._f_stride = [1, ] * self._num_levels
        self._bnorm = [False, ] + [True, ] * (self._num_levels - 2) + [False, ]
        self._res = [0, ] * self._num_levels
        self._bnorm_init_var = 1e-4
        self._bnorm_init_gamma = np.sqrt(2.0 / (9.0 * 64.0))
        self._bnorm_epsilon = 1e-5
        self._bnorm_decay = bnorm_decay

        self.level = [None, ] * self._num_levels
        self.input = images
        self.falg_train = falg_train
        self.extra_train = []
        self.variables_list = []
        self.trainable_list = []
        self.decay_list = []
        self.padding = padding

        x = self.input
        for i in range(self._num_levels):
            with tf.variable_scope('level_%d' % i):
                x = self._conv(x, self._f_size[i], self._f_num[i], self._f_stride[i], name='conv')
                if self._bnorm[i]:
                    x = self._batch_norm(x, name='bn')
                x = self._bias(x, name='bias')
                if self._res[i] > 0:
                    x = x + self.level[i - self._res[i]]
                x = self._actfun[i](x, name='active')
                self.level[i] = x
        self.output = x

    def _batch_norm(self, x, name='bnorm'):
        """Batch normalization."""
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]

            moving_mean = tf.get_variable(
                'moving_mean', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, dtype=tf.float32),
                trainable=False)
            moving_variance = tf.get_variable(
                'moving_variance', params_shape, tf.float32,
                initializer=tf.constant_initializer(self._bnorm_init_var, dtype=tf.float32),
                trainable=False)
            self.variables_list.append(moving_mean)
            self.variables_list.append(moving_variance)

            gamma = tf.get_variable(
                'gamma', params_shape, tf.float32,
                initializer=tf.random_normal_initializer(stddev=self._bnorm_init_gamma, dtype=tf.float32))
            self.variables_list.append(gamma)
            self.trainable_list.append(gamma)

            local_mean, local_variance = tf.nn.moments(x, [0, 1, 2], name='moments')

            mean, variance = tf.cond(
                self.falg_train, lambda: (local_mean, local_variance),
                lambda: (moving_mean, moving_variance))

            self.extra_train.append(moving_mean.assign_sub((1.0 - self._bnorm_decay) * (moving_mean - local_mean)))
            self.extra_train.append(
                moving_variance.assign_sub((1.0 - self._bnorm_decay) * (moving_variance - local_variance)))

            y = tf.nn.batch_normalization(
                x, mean, variance, None, gamma, self._bnorm_epsilon)
            y.set_shape(x.get_shape())
        return y

    def _bias(self, x, name='bias'):
        """Bias term."""
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]
            beta = tf.get_variable(
                'beta', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, dtype=tf.float32))
            self.variables_list.append(beta)
            self.trainable_list.append(beta)
            y = x + beta
        return y

    def _conv(self, x, filter_size, out_filters, stride, name='conv'):
        """Convolution."""
        with tf.variable_scope(name):
            in_filters = int(x.get_shape()[-1])
            n = filter_size * filter_size * np.maximum(in_filters, out_filters)
            kernel = tf.get_variable(
                'weights', [filter_size, filter_size, in_filters, out_filters],
                tf.float32, initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(2.0 / n), dtype=tf.float32))
            self.variables_list.append(kernel)
            self.trainable_list.append(kernel)
            self.decay_list.append(kernel)
            y = tf.nn.conv2d(x, kernel, [1, stride, stride, 1], padding=self.padding)
        return y


class NoiseprintEngine:
    tf.reset_default_graph()
    x_data = tf.placeholder(tf.float32, [1, None, None, 1], name="x_data")
    net = FullConvNet(x_data, 0.9, tf.constant(False), num_levels=17)
    saver = tf.train.Saver(net.variables_list)
    checkpoint_template = os.path.join(os.path.dirname(__file__), './noiseprint/net_jpg%d/model')
    configSess = tf.ConfigProto()
    configSess.gpu_options.allow_growth = True
    slide = 1024  # 3072
    largeLimit = 1050000  # 9437184
    overlap = 34

    def __init__(self, quality=101):
        self.quality = quality
        self.model = self.net
        self.session = None
        self.loaded_quality = None

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
        self.saver.restore(self.session, checkpoint)
        self.loaded_quality = quality

    def ensure_open(self):
        if self.loaded_quality is None:
            self.open()

    def predict_large(self, img):
        self.ensure_open()
        res = np.zeros((img.shape[0], img.shape[1]), np.float32)
        for index0 in range(0, img.shape[0], self.slide):
            index0start = index0 - self.overlap
            index0end = index0 + self.slide + self.overlap

            for index1 in range(0, img.shape[1], self.slide):
                index1start = index1 - self.overlap
                index1end = index1 + self.slide + self.overlap
                clip = img[max(index0start, 0): min(index0end, img.shape[0]), \
                       max(index1start, 0): min(index1end, img.shape[1])]
                resB = self.session.run(self.model.output, feed_dict={self.x_data: clip[np.newaxis, :, :, np.newaxis]})
                resB = np.squeeze(resB)

                if index0 > 0:
                    resB = resB[self.overlap:, :]
                if index1 > 0:
                    resB = resB[:, self.overlap:]
                resB = resB[:min(self.slide, resB.shape[0]), :min(self.slide, resB.shape[1])]

                res[index0: min(index0 + self.slide, res.shape[0]), \
                index1: min(index1 + self.slide, res.shape[1])] = resB
        return res

    def predict_small(self, img):
        self.ensure_open()
        res = self.session.run(self.model.output, feed_dict={self.x_data: img[np.newaxis, :, :, np.newaxis]})
        return np.squeeze(res)

    def predict(self, img):
        if img.shape[0] * img.shape[1] > self.largeLimit:
            return self.predict_large(img)
        else:
            return self.predict_small(img)

    def open(self, quality=None):
        self.session = tf.Session(config=self.configSess)
        if quality is None:
            quality = self.quality
        log("Opening noiseprint session with quality %d" % quality)
        self.quality = quality
        self.load_session(quality)

    def close(self):
        self.loaded_quality = None
        if self.session is not None:
            self.session.close()
        self.session = None
        log("Closing noiseprint session")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def gen_noiseprint(image, quality=None):
    if isinstance(image, str):
        if quality is None:
            quality = jpeg_qtableinv(image)
        image = imread2f_pil(image, channel=1)[0]
    else:
        if quality is None:
            quality = 101
    with NoiseprintEngine(quality) as engine:
        return engine.predict(image)

def normalize_noiseprint(noiseprint, margin=34):
    v_min = np.min(noiseprint[margin:-margin, margin:-margin])
    v_max = np.max(noiseprint[margin:-margin, margin:-margin])
    return ((noiseprint - v_min) / (v_max - v_min)).clip(0, 1)



