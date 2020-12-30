import tensorflow as tf
from tensorflow.python.data import Dataset
from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import Callback


class Metrics(Callback):

    def __init__(self, val_generator: Dataset):
        super().__init__()

        def filter_fn(i, n):
            zeros = [0] * n
            zeros[i] = 1
            return lambda x, y: tf.reduce_all(y == zeros)

        label_shape = val_generator.output_shapes[1]
        n = label_shape[0]

        self.cls = [val_generator.filter(filter_fn(i, n)).batch(128).cache() for i in range(n)]

    def on_epoch_end(self, epoch, logs=None):
        self.model: Model
        # print("L3", len(self.cls3))
        hand = getattr(self.model, "_eval_data_handler", None)
        self.model._eval_data_handler = None
        for i in range(len(self.cls)):
            print("Class %d:" % i)
            self.model.evaluate(self.cls[i])
        self.model._eval_data_handler = hand


class ParSine(tf.keras.layers.Layer):
    def __init__(self, w0: float = 1.0, **kwargs):
        """
        Sine activation function with w0 scaling support.

        Args:
            w0: w0 in the activation step `act(x; w0) = sin(w0 * x)`
        """
        super(ParSine, self).__init__(**kwargs)
        self.w0 = w0
        self.w1 = 1.0
        self.p = 0.5

    def call(self, inputs, **kwargs):
        return self.p * tf.sin(self.w1 * inputs + self.w0) + (1 - self.p) * inputs

    def get_config(self):
        config = {'w0': self.w0, 'w1': self.w1, 'p': self.p}
        base_config = super(ParSine, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
