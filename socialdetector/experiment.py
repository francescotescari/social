import os
from time import time

from tensorflow.python.data import Dataset
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.nadam import Nadam
from tensorflow_datasets.core import DatasetBuilder

from socialdetector.dataset_generator import full_dataset, tdfs_encode, Metrics
from socialdetector.dataset_utils import datasets_interleave, tf_print
from socialdetector.dl.model import GenericModel
from socialdetector.utility import is_listing


class DatasetSpec:

    def __init__(self, path_datasets, noiseprint_dir, origin_dir, files_number=None, pixel_estimate=None):
        self.path_datasets = path_datasets
        self.noiseprint_path = noiseprint_dir
        self.origin_path = origin_dir
        self.files_number = files_number
        self.pixel_estimate = pixel_estimate

    def label_path_gen_as_dict(self):
        out = {}
        for folder, label in self.path_datasets:
            if label not in out:
                out[label] = []
            out[label].append(folder)
        return out

    def estimated_batches(self, batch_size, tile_size=(64, 64)):
        return round(self.pixel_estimate / (tile_size[0] * tile_size[1]) / batch_size)

    def __repr__(self):
        raise NotImplementedError


class DatasetSplitter:

    def split(self, dataset):
        raise NotImplementedError


class DatasetConstructor:
    batch_size = [256, 256, 256]
    shuffle = 2000
    seed = round(time())

    noiseprint = False
    dct_encoding = None
    same_seed = False
    print_log = print

    def _split(self, dataset_spec, splitter: DatasetSplitter):
        noiseprint_path, origin_path = (None, None) if not self.noiseprint else (dataset_spec.noiseprint_path,
                                                                                 dataset_spec.origin_path)
        dct_encoding = self.dct_encoding
        different_labels_ds = []
        i = 0
        for label, path_datasets in dataset_spec.label_path_gen_as_dict().items():
            if isinstance(label, tuple):
                label = list(label)
            ds = datasets_interleave(path_datasets)
            l_seed = self.seed if self.same_seed else self.seed * 37 + i * 13
            parts_datasets = list(splitter.split(ds))
            if self.print_log is not None:
                self.print_log("Validation files:")
                self.print_log(list(parts_datasets[1].as_numpy_iterator()))
                self.print_log("Test files:")
                self.print_log(list(parts_datasets[2].as_numpy_iterator()))

            if self.shuffle > 0:
                parts_datasets = [
                    ds.shuffle(2 * dataset_spec.files_number, seed=l_seed, reshuffle_each_iteration=True) for ds in
                    parts_datasets
                ]

            # parts_datasets[0] = parts_datasets[0].map(lambda x: tf_print(x))
            parts_datasets = [
                full_dataset(ds, noiseprint_path=noiseprint_path, dct_encoding=dct_encoding, origin_path=origin_path)
                for ds in parts_datasets
            ]
            if self.shuffle > 0:
                parts_datasets = [
                    ds.shuffle(self.shuffle, seed=l_seed, reshuffle_each_iteration=True) for ds in parts_datasets
                ]

            label_ds = Dataset.from_tensors(label).repeat()
            parts_datasets = [Dataset.zip((ds, label_ds)) for ds in parts_datasets]
            different_labels_ds.append(parts_datasets)
            i += 1

        return [datasets_interleave([d[i] for d in different_labels_ds]) for i in range(3)]

    def get_split(self, dataset_spec: DatasetSpec, splitter: DatasetSplitter):
        res = self._split(dataset_spec, splitter)
        return [res[i].batch(self.batch_size[i]) if self.batch_size[i] > 0 else res[i] for i in range(3)]

    def split_from_builder(self, dataset_builder: DatasetBuilder):
        datasets = dataset_builder.as_dataset()
        names = ['train', 'validation', 'test']
        encoded = [tdfs_encode(datasets[name], noiseprint=self.noiseprint, dct_encoding=self.dct_encoding,
                               shuffle=(self.shuffle if name == 'train' else 0)) for name in names]
        # encoded = [ds.shuffle() for ds in encoded]
        return [encoded[i].batch(self.batch_size[i]) if self.batch_size[i] > 0 else encoded[i] for i in range(3)]


def require_not_none(var, msg):
    if var is None:
        raise ValueError(msg)




class Experiment:
    loss_function = 'categorical_crossentropy'
    metric_functions = ['accuracy']
    optimizer = Nadam(lr=0.0001)
    steps_per_epoch = 1000

    repeat_train = True
    ds_splitter = None
    model_type = None
    dataset_spec: DatasetSpec = None
    dataset_builder = None

    def __repr__(self):
        raise NotImplementedError

    def __init__(self):
        self.compile_config = {}
        self.train_config = {}
        self.model = None
        self.ds_constructor = DatasetConstructor()
        self.split_ds = None

    def _prepare_configs(self):
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
        self._assure_model_ready()
        dss = self.get_datasets()
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
            self.model.build_model()
            self.model.compile(**self.compile_config)
        self.model.id = os.path.join(repr(self), repr(self.dataset_spec))
        print("Model %s ready" % self.model.id)

    def _assure_generators_ready(self):
        if self.split_ds is not None:
            return
        if self.dataset_builder is None:
            require_not_none(self.dataset_spec, "No dataset for this experiment")
            require_not_none(self.ds_splitter, "No splitter for the dataset")
            print("Generating train, validation and test datasets...")
            self.split_ds = list(self.ds_constructor.get_split(self.dataset_spec, self.ds_splitter))
        else:
            self.split_ds = list(self.ds_constructor.split_from_builder(self.dataset_builder))


empty_dataset = Dataset.from_tensors([])


class AllTrainSplitter(DatasetSplitter):

    def split(self, dataset):
        return dataset, empty_dataset, empty_dataset


class FilterSplitter(DatasetSplitter):

    def __init__(self, val_filter, tst_filter):
        super().__init__()
        self.val_filter = val_filter
        self.tst_filter = tst_filter

    def split(self, dataset):
        train = dataset.filter(lambda x: not self.val_filter(x) and not self.tst_filter(x)).cache()
        val = dataset.filter(self.val_filter).cache()
        test = dataset.filter(self.tst_filter).cache()
        return train, val, test


class FixedSizeSplitter(DatasetSplitter):

    def __init__(self, val_size, test_size, shuffle=10000, seed=12321):
        super().__init__()
        self.val_size = val_size
        self.test_size = test_size
        self.shuffle = shuffle
        self.seed = seed if seed is not None else round(time())
        self.seed_incr = 0

    def split(self, dataset):
        if self.shuffle > 0:
            dataset = dataset.shuffle(self.shuffle, seed=self.seed + self.seed_incr * 101,
                                      reshuffle_each_iteration=False)
        self.seed_incr += 43
        train = dataset.skip(self.val_size + self.test_size).cache()
        val = dataset.take(self.val_size).cache().cache()
        tst = dataset.skip(self.val_size).take(self.test_size).cache()
        return train, val, tst
