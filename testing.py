from tensorflow.python.keras import Model
from tensorflow.python.keras.metrics import Recall

import socialdetector.tf_options_setter
from experiments import ExpNoiseprint, ExpMyJpeg, ExpPaperJpeg, ExpTwoStreams
from socialdetector.dataset.social_images import UcidPublic, UcidSocial, IpLabSeven, IpLabThree
from socialdetector.train_utils import EvaluateCallback


def test_noiseprintonly():
    path = "./history/noiseprint_only/ucid_public/1609511639.9576898_std/e199-l0.2542-v0.4447.h5"

    experiment = ExpNoiseprint()
    experiment.dataset_builder = UcidPublic()
    experiment.load_from(path).evaluate()


def test_myjpeg():
    path = "./history/my_jpeg/ucid_public/1609359679.82444_long/e220-l0.2023-v0.3521.ckpt"
    experiment = ExpMyJpeg()
    experiment.dataset_builder = UcidPublic()
    experiment.load_from(path).evaluate()


def test_paperjpeg():
    path = "./history/jpeg_paper/1608074487/e39-l0.1786-v0.1852.h5"
    path = "./history/jpeg_paper/1608074487/e35-l0.1290-v0.2095.h5"
    experiment = ExpPaperJpeg()
    experiment.dataset_builder = UcidSocial()
    experiment.load_from(path).evaluate()


def test_twostream():
    path = "./history/two_streams/ucid_public/1609540064.0625489_std/e230-l0.0450-v0.3101.h5"
    #path = ".\\history\\two_streams\\ucid_social\\1609585022.0416644_std\\e51-l0.0184-v0.0287.h5"
    path = r".\history\ucid_public\two_streams_999999\1609679999.9570293_std\e29-l0.2712-v0.3808.h5"
    experiment = ExpTwoStreams()
    experiment.dataset_builder = UcidPublic()
    experiment.load_from(path)

    experiment.evaluate(True)


test_twostream()
