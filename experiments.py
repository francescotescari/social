from abc import ABC

from socialdetector.dataset_generator import encode_coefficients_my, encode_coefficients_paper
from socialdetector.dl.jpeg_cnn import MyCNNJpeg, PaperCNNModel
from socialdetector.dl.noiseprint_cnn import NoiseprintModel, FullModel
from socialdetector.experiment import Experiment


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

    def __init__(self, noiseprint=False, dct_encoding=None):
        super().__init__()
        self.noiseprint = noiseprint
        self.dct_encoding = dct_encoding


class ExpNoiseprint(StdLocalExperiment):
    model_type = NoiseprintModel
    name = "noiseprint_only"

    def __init__(self):
        super().__init__(noiseprint=True)


class ExpMyJpeg(StdLocalExperiment):
    model_type = MyCNNJpeg
    name = "my_jpeg"

    def __init__(self):
        super().__init__(dct_encoding=encode_coefficients_my)


class ExpPaperJpeg(StdLocalExperiment):
    model_type = PaperCNNModel
    name = "jpeg_paper"

    def __init__(self):
        super().__init__(dct_encoding=encode_coefficients_paper)


class ExpTwoStreams(StdLocalExperiment):
    model_type = FullModel
    name = "two_streams"

    def __init__(self):
        super().__init__(dct_encoding=encode_coefficients_my, noiseprint=True)
