import os
from time import time

from PIL import Image
from tensorflow.python.keras.metrics import Recall
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.nadam import Nadam

from socialdetector.dataset.social_images.social_images import SocialImages
from socialdetector.train_utils import Metrics, ConfusionMatrix, EvaluateCallback, MyValidation, balance_validation

from socialdetector.dl.model import GenericModel
from socialdetector.ds_split import DsSplit
from socialdetector.utility import jpeg_quality_of


def require_not_none(obj, msg):
    if obj is None:
        raise ValueError(msg)


class Experiment:
    loss_function = 'categorical_crossentropy'
    metric_functions = [Recall(), 'accuracy']
    optimizer = Adam(lr=0.001)
    steps_per_epoch = None

    repeat_train = True
    ds_splitter = None
    model_type = None
    dataset_builder: SocialImages = None
    extra = None
    batch_size = 256
    noiseprint = False
    dct_encoding = None
    default_steps = 1000
    shuffle = 200000
    seed = 1302

    def __repr__(self):
        raise NotImplementedError

    def __init__(self):
        self.compile_config = {}
        self.train_config = {}
        self.model = None
        self.split_ds = None
        self.splitter = None

    def _prepare_configs(self):
        if self.steps_per_epoch is None:
            self.steps_per_epoch = self.default_steps

        self.compile_config.update({
            'loss_function': self.loss_function,
            'metric_functions': self.metric_functions,
            'optimizer': self.optimizer
        })
        self.train_config.update({
            'steps_per_epoch': self.steps_per_epoch,
            'class_weight': None if self.splitter is None else self.splitter.class_weights
        })

    @property
    def img_mappings(self):
        self._assure_generators_ready()
        return self.splitter.mappings

    @property
    def labels_map(self):
        self._assure_generators_ready()
        return self.splitter.labels_map

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
        tst = dss[2].cache()
        # self.model.registered_callbacks.append(Metrics(val, 2))
        #self.model.registered_callbacks.insert(0, MyValidation(val, 256, name='Validation'))
        #self.model.registered_callbacks.insert(1, MyValidation(tst, 256, name='Test'))
        val = balance_validation(val).batch(self.batch_size).cache()
        self.model.train_with_generator(train, epochs=20000, validation_data=val, **self.train_config)

    def evaluate(self, full_evaluation=False):
        self._assure_model_ready()
        test = self.get_datasets()[2]
        model = self.model.model
        matrix = ConfusionMatrix(self.splitter.labels)
        model.compile(optimizer=model.optimizer,
                      loss=model.loss,
                      metrics=[Recall(), matrix])
        #self.model.model.evaluate(test.batch(256))
        matrix.print_result()
        if full_evaluation:
            EvaluateCallback.run_experiment(self)


    def predict(self, image_path, noiseprint_path=None):
        """image = Image.open(image_path)
        quality = jpeg_quality_of(image)
        image = np.asarray(image.convert('YCbCr'))[..., 0]"""

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
        self.model.id = os.path.join("None" if self.dataset_builder is None else self.dataset_builder.name,
                                     repr(self) + "_" + str(self.seed))
        self.model.desc = self.extra
        print("Model %s ready" % self.model.id)

    def _assure_generators_ready(self):
        if self.split_ds is not None:
            return
        require_not_none(self.dataset_builder, "No dataset for this experiment")
        self.splitter = DsSplit(noiseprint=self.noiseprint, dct_encoding=self.dct_encoding, seed=self.seed,
                                shuffle_train=self.shuffle)

        res = list(self.splitter.split_datasets(self.dataset_builder))
        self.default_steps = self.splitter.min_chunks // self.batch_size // 4
        if self.batch_size > 0:
            res[0] = res[0].batch(self.batch_size)

        self.split_ds = res
        # print("MAP", self.img_mappings)
