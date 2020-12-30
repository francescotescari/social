import os
from time import time

from tensorflow.python.keras.optimizer_v2.nadam import Nadam

from socialdetector.dataset.social_images.social_images import SocialImages
from socialdetector.train_utils import Metrics

from socialdetector.dl.model import GenericModel
from socialdetector.ds_split import DsSplit


class DatasetConstructor:
    batch_size = [256, 256, 256]
    shuffle = 5000
    seed = round(time())

    noiseprint = False
    dct_encoding = None
    _default_steps = 1000

    def split_from_builder(self, dataset_builder: SocialImages):
        splitter = DsSplit(noiseprint=self.noiseprint, dct_encoding=self.dct_encoding, seed=self.seed,
                           shuffle_train=self.shuffle)

        res = splitter.split_datasets(dataset_builder)
        self._default_steps = splitter.min_chunks // 4
        if self.batch_size[0] > 0:
            self._default_steps = self._default_steps // self.batch_size[0]
        return [res[i].batch(self.batch_size[i]) if i == 0 else res[i] for i in range(3)]

    @property
    def default_steps(self):
        return self._default_steps


def require_not_none(var, msg):
    if var is None:
        raise ValueError(msg)


class Experiment:
    loss_function = 'categorical_crossentropy'
    metric_functions = ['accuracy']
    optimizer = Nadam(lr=0.0001)
    steps_per_epoch = None

    repeat_train = True
    ds_splitter = None
    model_type = None
    dataset_builder: SocialImages = None
    extra = None
    batch_size = 512

    def __repr__(self):
        raise NotImplementedError

    def __init__(self):
        self.compile_config = {}
        self.train_config = {}
        self.model = None
        self.ds_constructor = DatasetConstructor()
        self.split_ds = None

    def _prepare_configs(self):
        if self.steps_per_epoch is None:
            self.steps_per_epoch = self.ds_constructor.default_steps

        self.compile_config.update({
            'loss_function': self.loss_function,
            'metric_functions': self.metric_functions,
            'optimizer': self.optimizer
        })
        self.train_config.update({
            'steps_per_epoch': self.steps_per_epoch
        })

    def load_from(self, path):
        if path is not None:
            self.model = GenericModel.load_from(path)
        return self

    def get_datasets(self):
        self._assure_generators_ready()
        return self.split_ds

    def train(self):
        dss = self.get_datasets()
        self._assure_model_ready()
        train = dss[0]
        if self.repeat_train:
            train = train.repeat()
        val = dss[1].cache()
        self.model.registered_callbacks.append(Metrics(val))
        self.model.train_with_generator(train, epochs=20000, validation_data=val.batch(256), **self.train_config)

    def evaluate(self):
        self._assure_model_ready()
        test = self.get_datasets()[2]
        self.model.model.evaluate(test)

    def _assure_model_ready(self):
        self._prepare_configs()
        if self.model is None:
            print("Compiling new initialized model %r" % self.model_type)
            require_not_none(self.model_type, "No model type specified")
            self.model: GenericModel = self.model_type()
            if self.dataset_builder is not None:
                self.model.classes = len(self.dataset_builder.labels)
            self.model.build_model()
            self.model.compile(**self.compile_config)
        self.model.id = os.path.join(repr(self), "None" if self.dataset_builder is None else self.dataset_builder.name)
        self.model.desc = self.extra
        print("Model %s ready" % self.model.id)

    def _assure_generators_ready(self):
        if self.split_ds is not None:
            return
        self.ds_constructor.batch_size = [self.batch_size, self.batch_size, self.batch_size]
        require_not_none(self.dataset_builder, "No dataset for this experiment")
        self.split_ds = list(self.ds_constructor.split_from_builder(self.dataset_builder))
