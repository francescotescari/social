import os
from typing import List

import tensorflow as tf
from tensorflow.python.data import Dataset
from tensorflow.python.data.experimental import group_by_window
from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.metrics import Recall, Metric, CategoricalCrossentropy
import numpy as np
from texttable import Texttable

from socialdetector.dataset.social_images.social_images import SocialImages
from socialdetector.dataset_utils import datasets_concatenate
from socialdetector.ds_split import DsSplit


class Metrics(Callback):

    def __init__(self, val_generator: Dataset, freq=10):
        super().__init__()

        def filter_fn(i, n):
            zeros = [0] * n
            zeros[i] = 1
            return lambda *args: tf.reduce_all(args[1] == zeros)

        label_shape = val_generator.output_shapes[1]
        n = label_shape[0]

        self.freq = freq
        self.max_met = 0

        self.cls = [val_generator.filter(filter_fn(i, n)).batch(128).cache() for i in range(n)]

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.freq != 1:
            return
        self.model: Model
        # print("L3", len(self.cls3))
        hand = getattr(self.model, "_eval_data_handler", None)
        self.model._eval_data_handler = None
        mets = []
        for i in range(len(self.cls)):
            print("Class %d:" % i)
            mets.append(self.model.evaluate(self.cls[i]))
        self.model._eval_data_handler = hand
        mets = np.mean(np.array(mets), axis=0)
        print("\n\nMetrics: ", mets)
        val = mets[1]
        if val > self.max_met:
            self.max_met = val
            print("NEW_MAX", val)


class MyValidation(Callback):

    def __init__(self, ds, batch_size=128, name='Validation'):

        super().__init__()

        def filter_fn(i, n):
            zeros = [0] * n
            zeros[i] = 1
            return lambda *args: tf.reduce_all(args[1] == zeros)

        label_shape = tf.compat.v1.data.get_output_shapes(ds)[1]
        n = label_shape[0]

        self.max_met = 0
        self.name = name
        self.cls = [ds.filter(filter_fn(i, n)).batch(batch_size).cache() for i in range(n)]
        self.fp = open('log_%s.txt' % name, 'w')

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        hand = getattr(self.model, "_eval_data_handler", None)
        self.model._eval_data_handler = None
        mets = []
        print('\n')
        for i in range(len(self.cls)):
            print("%s %d:" % (self.name, i))
            mets.append(self.model.evaluate(self.cls[i]))
        self.model._eval_data_handler = hand
        mets = np.mean(np.array(mets), axis=0)
        self.model: Model
        assert len(self.model.metrics) == (len(mets))
        for m, val in zip(self.model.metrics, mets):
            m: Metric
            logs['val_' + m.name] = val
        print(logs)
        self.fp.write('%d: %r\n' % (epoch, logs))
        self.fp.flush()
        return logs


def evaluate_ds(model, ds, mapping, img_metrics=(), chunk_metrics=()):
    def take(ds, n):
        pred, true = [], []
        for i in range(n):
            e = next(ds)
            pred.append(e[0])
            true.append(e[1])
        return pred, true

    def map_pred(ds):
        for sample in ds:
            x = sample[0]
            y_true = sample[1]
            y_pred = model.predict(x)
            for e in zip(y_pred, y_true):
                yield e

    def form(arr):
        return list(map('{:.2f}'.format, arr))

    n_lab = tf.compat.v1.data.get_output_shapes(ds)[1][1]
    ds = map_pred(ds)
    cce = CategoricalCrossentropy()

    total = 0
    avgs = []
    for entry in mapping:
        name, chunks = entry
        name = os.path.basename(name.decode("utf-8"))
        y_pred, y_true = take(ds, chunks)
        total += len(y_pred)
        # print(name, chunks, "LOSS", cce(y_true, y_pred).numpy())
        for metric in chunk_metrics:
            metric.update_state(np.array(y_true), np.array(y_pred))
        y_pred2 = np.mean(y_pred, axis=0)
        y_pred = np.argmax(y_pred, axis=1)
        zeros = [0] * n_lab
        ele, count = np.unique(y_pred, return_counts=True)
        lbm = {i: 0 for i in range(n_lab)}
        lbm = {**lbm, **{ele[i]: count[i] for i in range(len(ele))}}
        counts = [int(lbm[i]) for i in range(n_lab)]
        y_pred = ele[max(range(len(ele)), key=lambda x: count[x])]
        zeros[y_pred] = 1
        y_pred = zeros
        y_true = y_true[0]
        print(counts[tf.argmax(y_true)], '/', sum(counts))
        avgs.append([n / sum(counts) for n in counts])
        if tf.argmax(y_true) != tf.argmax(y_pred):
            print("WRONG: %s" % name, form(y_pred2), " | ", counts, " -> ", form(y_true))
        # print(form(y_pred), form(y_true.numpy()))
        for metric in img_metrics:
            metric.update_state(y_true, y_pred)

    more = 0
    print("AVGSS", np.mean(np.array(avgs), axis=0))
    try:
        while True:
            next(ds)
            more += 1
    except StopIteration:
        if more:
            print("WARNING: more data then expected (%d)" % more)

    print("Evaluated %d/%d chunks" % (total, total + more))


class EvaluateCallback(Callback):

    @staticmethod
    def run_experiment(exp):
        exp.get_datasets()
        e = EvaluateCallback(exp.get_datasets()[2], exp.img_mappings[2], exp.labels_map)
        e.model = exp.model.model
        e.eval()

    @staticmethod
    def run(model, *args, **kwargs):
        cb = EvaluateCallback(*args, **kwargs)
        cb.model = model
        cb.eval()

    def __init__(self, val_generator: Dataset, mapping, labels_map):
        super().__init__()

        inverse_labels = {tuple(v): k for k, v in labels_map.items()}

        def label_of(i, n):
            zeros = [0] * n
            zeros[i] = 1
            return zeros

        def filter_fn(i, n):
            lb = label_of(i, n)
            return lambda *args: tf.reduce_all(args[1] == lb)

        label_shape = tf.compat.v1.data.get_output_shapes(val_generator)[1]
        n = label_shape[0]

        self.cls = [val_generator.filter(filter_fn(i, n)).batch(128).cache() for i in range(n)]
        self.labels = [inverse_labels[tuple(label_of(i, n))] for i in range(n)]
        self.mappings = [mapping[lb] for lb in self.labels]

    def on_epoch_end(self, epoch, logs=None):
        self.eval()

    def eval(self):
        image_wise = Recall()
        chunk_wise = Recall()
        cm = ConfusionMatrix(self.labels)
        chunk_cm = ConfusionMatrix(self.labels)
        for cl_ds, mapping, label in zip(self.cls, self.mappings, self.labels):
            class_wise = Recall()
            class_chunk = Recall()
            evaluate_ds(self.model, cl_ds, mapping, (image_wise, class_wise, cm), (chunk_cm, chunk_wise, class_chunk))
            print("Image class recall %s" % label)
            print(class_wise.result())
            print("Chunk class recall %s" % label)
            print(class_chunk.result())
        print("Image-wise recall")
        print(image_wise.result())
        print("Chunk-wise recall")
        print(chunk_wise.result())
        print("Confusion matrix")
        cm.print_result()
        print("Chunk matrix")
        chunk_cm.print_result()


class ConfusionMatrix(Metric):

    def __init__(self, labels, **kwargs):
        super().__init__(**kwargs)
        self.labels = labels
        self.size = len(labels)
        self.matrix = [[0] * self.size for _ in range(self.size)]

    def update_state_np(self, y_true, y_pred):
        if len(y_true.shape) > 0:
            for t, p in zip(y_true, y_pred):
                self.update_state_np(t, p)
            return
        self.matrix[y_true][y_pred] += 1

    def get_matrix(self):
        return self.matrix

    def update_state(self, y_true, y_pred, sample_weight=None):
        if len(y_true.shape) > 1:
            true_cls = tf.argmax(y_true, axis=1)
            pred_cls = tf.argmax(y_pred, axis=1)
            return tf.numpy_function(self.update_state_np, [true_cls, pred_cls], ())
        true_cls = tf.argmax(y_true)
        pred_cls = tf.argmax(y_pred)
        self.matrix[true_cls][pred_cls] += 1

    def result(self):
        mat = tf.numpy_function(self.get_matrix, (), [tf.int32, tf.int32, tf.int32])
        # mat.set_shape((self.size, self.size))
        """for pred_line in self.matrix:
            tot = sum(pred_line)
            if tot > 0:
                mat.append([x / tot for x in pred_line])
            else:
                mat.append(pred_line)"""
        # tf.print(mat)
        # tf.print(self.matrix)
        return mat

    def print_result(self):
        table = Texttable()
        res = self.result().numpy().reshape((self.size, self.size))
        th = ["", *self.labels]
        table.add_row(th)
        i = 0
        for pred_line in res:
            tot = sum(pred_line)
            if tot > 0:
                pred_line = [x / tot for x in pred_line]
            table.add_row([self.labels[i], *pred_line])
            i += 1

        print(table.draw())

def _all_equals(iterator):
    first = True
    last = None
    for i in iterator:
        if not first and i != last:
            return False
        first = False
        last = i
    return True

def balance_validation(ds):
    print('Balancing validation...')

    def filter_fn(i, n):
        zeros = [0] * n
        zeros[i] = 1
        return lambda *args: tf.reduce_all(args[1] == zeros)

    oss = tf.compat.v1.data.get_output_shapes(ds)
    label_shape = oss[1]
    n = label_shape[0]

    if len(oss) > 2:
        gw = lambda *args: args[2]
        st = lambda f: lambda x, y, w: (x, y, w*f)
    else:
        gw = lambda *args: 1
        st = lambda f: lambda x, y: (x, y, f)

    cls = [ds.filter(filter_fn(i, n)) for i in range(n)]
    weights = [sum(cl.map(gw).as_numpy_iterator()) for cl in cls]
    print('Before balancing', weights)
    if _all_equals([round(w*100) for w in weights]):
        return ds
    dest_w = np.mean(weights)
    fs = [dest_w/w for w in weights]
    fns = [st(f) for f in fs]
    cls = [cl.map(f) for cl, f in zip(cls, fns)]
    weights = [sum(cl.map(lambda *args: args[2]).as_numpy_iterator()) for cl in cls]
    print('After balancing', weights)
    return datasets_concatenate(cls)

