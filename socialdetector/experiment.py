from time import time

from tensorflow.python.keras.optimizer_v2.adam import Adam

from socialdetector.dataset_generator import Splittable
from socialdetector.dl.model import GenericModel







class Experiment:

    def __init__(self, model_type, dataset_generator: Splittable):
        self.train_gen = None
        self.validation_gen = None
        self.test_gen = None
        self.all_gen = dataset_generator
        self.model_type = model_type
        self.model = None
        self.model_id = None
        self.dataset_config = {}
        self.compile_config = {
            'loss_function': 'mse',
            'metric_functions': ['accuracy'],
            'optimizer': Adam(lr=0.001)
        }
        self.train_config = {
            'steps_per_epoch': 0,
        }

    def load_from(self, path):
        if path is not None:
            self.model = GenericModel.load_from(path)
        return self

    def change_dataset_generator(self, dataset_generator: Splittable):
        self.train_gen = None
        self.validation_gen = None
        self.test_gen = None
        self.all_gen = dataset_generator

    def train(self):
        self._assure_model_ready()
        self._assure_generators_ready()
        self.model.train_with_generator(self.train_gen.repeat(), epochs=20000, validation_data=self.validation_gen,
                                        **self.train_config)

    def evaluate(self):
        self._assure_model_ready()
        self._assure_generators_ready()
        self.model.model.evaluate(self.test_gen)

    def __assure_model_id(self):
        if self.model_id is not None:
            return
        assert self.model is not None
        self.model_id = "%s_%d" % (str(self.__class__.__name__).lower(), round(time()))
        self.model.id = self.model_id

    def _assure_model_ready(self):
        if self.model is None:
            print("Compiling new initialized model %r" % self.model_type)
            self.model: GenericModel = self.model_type()
            self.model.build_model()
            self.model.compile(**self.compile_config)
        self.__assure_model_id()
        print("Model %s ready" % self.model_id)

    def _assure_generators_ready(self):
        if self.train_gen is not None:
            return
        print("Generating train, validation and test datasets...")
        self.train_gen, self.validation_gen, self.test_gen = self.all_gen.split_build(**self.dataset_config)
