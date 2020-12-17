from abc import ABC

from tensorflow.python.keras.optimizer_v2.adam import Adam

from socialdetector.dataset_generator import encode_coefficients_my, encode_coefficients_paper
from socialdetector.dl.jpeg_cnn import MyCNNJpeg, PaperCNNModel
from socialdetector.dl.noiseprint_cnn import NoiseprintModel
from socialdetector.experiment import Experiment, FixedSizeSplitter


class LocalExperiment(Experiment, ABC):
    tensorboard_folder = "./tsboard"
    history_folder = "./history"
    registered_already = False

    def _assure_model_ready(self):
        super()._assure_model_ready()
        if self.registered_already:
            return
        self.registered_already = True
        self.model.register_std_callbacks(tensorboard_logs_folder=self.tensorboard_folder,
                                          checkpoint_path=self.history_folder)


class StdLocalExperiment(LocalExperiment):
    name = None

    def __repr__(self):
        if self.name is not None:
            return self.name
        return self.__class__.__name__.lower()

    batch_size = 256

    def __init__(self, noiseprint=False, dct_encoding=None):
        super().__init__()
        self.ds_constructor.shuffle = 2000
        self.ds_constructor.seed = 12321
        self.ds_constructor.dct_encoding = dct_encoding
        self.ds_constructor.noiseprint = noiseprint

    def _prepare_configs(self):
        if self.dataset_spec is not None:
            self.steps_per_epoch = round(self.dataset_spec.estimated_batches(self.batch_size) * 0.8 / 2)
        return super()._prepare_configs()

    def get_datasets(self):
        self.ds_constructor.batch_size = [self.batch_size, self.batch_size, self.batch_size]
        if self.ds_splitter is None and self.dataset_spec is not None:
            size = self.dataset_spec.files_number
            self.ds_splitter = FixedSizeSplitter(size // 10, size // 10, shuffle=size * 2)
        return super().get_datasets()


class NoiseprintOnly(StdLocalExperiment):
    model_type = NoiseprintModel
    name = "noiseprint_only"

    def __init__(self):
        super().__init__(noiseprint=True)


class MyJpeg(StdLocalExperiment):
    model_type = MyCNNJpeg
    optimizer = Adam(lr=0.00005)
    name = "my_jpeg"

    def __init__(self):
        super().__init__(dct_encoding=encode_coefficients_my)


class JpegPaper(StdLocalExperiment):
    model_type = PaperCNNModel
    name = "jpeg_paper"

    def __init__(self):
        super().__init__(dct_encoding=encode_coefficients_paper)
