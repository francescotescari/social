import os
from time import time

import numpy as np
from tensorflow.core.protobuf.config_pb2 import ConfigProto
from tensorflow.python.client.session import InteractiveSession
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.optimizer_v2.adam import Adam


class DumpValidationPredictions(Callback):

    def __init__(self, data):
        self.data = data
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        predicted = np.around(self.model.predict(self.data[0])[0:10], 2)
        print("\n", predicted, "\n", self.data[1][0:10])


class GenericModel:

    @staticmethod
    def load_from(path):
        model = GenericModel()
        model.model = load_model(path)
        return model

    def __init__(self):
        self.model = None
        self.registered_callbacks = []
        self.id = 'generic_model'
        self.time = round(time())
        """config = ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.40
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)"""

    def build_model(self):
        img_input = Input(self.get_input_shape())
        last_layer = self.model_structure(img_input)
        self.model = Model(img_input, last_layer)
        self.model.summary()

    def compile(self, loss_function, metric_functions=None, optimizer=Adam(1e-3, epsilon=1e-6)):
        self.require_model_loaded()
        return self.model.compile(loss=loss_function, optimizer=optimizer, metrics=metric_functions)

    def model_structure(self, input_img):
        raise NotImplementedError

    def get_input_shape(self):
        raise NotImplementedError

    def register_std_callbacks(self, tensorboard_logs_folder=None, checkpoint_path=None):
        self.require_model_loaded()
        folder_id = os.path.join(self.id, str(self.time))
        if tensorboard_logs_folder is not None:
            self.registered_callbacks.append(
                TensorBoard(log_dir=os.path.join(tensorboard_logs_folder, folder_id), histogram_freq=0, write_graph=True,
                            write_images=True))

        if checkpoint_path is not None:
            store_path = os.path.join(checkpoint_path, folder_id)
            if not os.path.exists(store_path):
                os.makedirs(store_path)
            store_path = os.path.join(store_path, 'e{epoch:02d}-l{loss:.4f}-v{val_loss:.4f}.ckpt')
            print("Storing to %s" % store_path)
            self.registered_callbacks.append(
                ModelCheckpoint(store_path, monitor='val_loss', verbose=1, period=1, save_best_only=False, mode='min'))

    def train_with_generator(self, training_data_generator, epochs,
                             steps_per_epoch, validation_data=None):
        self.model.fit(training_data_generator,
                       use_multiprocessing=True, workers=4, steps_per_epoch=steps_per_epoch,
                       callbacks=self.registered_callbacks, epochs=epochs, verbose=1,
                       **({} if validation_data is None else {"validation_data": validation_data}))

    def require_model_loaded(self):
        if self.model is None:
            raise ValueError("Model is not build yet")

    def load_weights(self, path):
        self.require_model_loaded()
        return self.model.load_weights(path)

    def predict(self, batch):
        self.require_model_loaded()
        return self.model.predict(batch)
