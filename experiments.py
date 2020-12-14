from data_loader import ucid_social, estimated_len
from socialdetector.dataset_generator import encode_coefficients_my, encode_coefficients_paper
from socialdetector.dl.jpeg_cnn import MyCNNJpeg, PaperCNNModel
from socialdetector.dl.noiseprint_cnn import NoiseprintModel
from socialdetector.experiment import Experiment


class LocalExperiment(Experiment):
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


class NoiseprintOnly(LocalExperiment):
    batch_size = 256

    def __init__(self, dataset):
        super().__init__(NoiseprintModel, dataset)
        self.train_config["steps_per_epoch"] = estimated_len(ucid_social) // self.batch_size // 2
        self.dataset_config["batch_size"] = self.batch_size
        self.dataset_config["noiseprint"] = True


class MyJpeg(LocalExperiment):
    batch_size = 256

    def __init__(self, dataset):
        super().__init__(MyCNNJpeg, dataset)
        self.train_config["steps_per_epoch"] = estimated_len(ucid_social) // self.batch_size // 2
        self.dataset_config["batch_size"] = self.batch_size
        self.dataset_config["encoding"] = encode_coefficients_my


class JpegPaper(LocalExperiment):
    batch_size = 256

    def __init__(self, dataset):
        super().__init__(PaperCNNModel, dataset)
        self.train_config["steps_per_epoch"] = estimated_len(ucid_social) // self.batch_size // 2
        self.dataset_config["batch_size"] = self.batch_size
        self.dataset_config["encoding"] = encode_coefficients_paper
