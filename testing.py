from tensorflow.python.keras import Model
from tensorflow.python.keras.metrics import Recall

import socialdetector.tf_options_setter
from experiments import ExpNoiseprint, ExpMyJpeg, ExpPaperJpeg, ExpTwoStreams
from socialdetector.dataset.social_images import UcidPublic, UcidSocial, IpLabSeven, IpLabThree
from socialdetector.train_utils import EvaluateCallback


def test_noiseprintonly():
    path = r".\history\ucid_public\noiseprint_only_12321\1609707570.874374_std\e07-l0.3366-v0.5750.h5"

    experiment = ExpNoiseprint()
    experiment.dataset_builder = UcidPublic()
    experiment.load_from(path).evaluate(True)


def test_myjpeg():
    path = r".\history\ucid_public\jpeg_paper_12321\1609704892.370081_std\e20-l0.2306-v0.4067.h5"
    experiment = ExpMyJpeg()
    experiment.dataset_builder = UcidPublic()
    experiment.load_from(path).evaluate(True)


def test_paperjpeg():
    path = "./history/jpeg_paper/1608074487/e39-l0.1786-v0.1852.h5"
    path = r".\history\ucid_public\jpeg_paper_12321\1609704892.370081_std\e20-l0.2306-v0.4067.h5"
    experiment = ExpPaperJpeg()
    experiment.dataset_builder = UcidPublic()
    experiment.load_from(path).evaluate(True)


def test_twostream():
    path = "./history/two_streams/ucid_public/1609540064.0625489_std/e230-l0.0450-v0.3101.h5"
    #path = ".\\history\\two_streams\\ucid_social\\1609585022.0416644_std\\e51-l0.0184-v0.0287.h5"
    path = r".\history\ucid_public\two_streams_12321\1609709473.1455812_std\e23-l0.1074-v0.4027.h5"
    experiment = ExpTwoStreams()
    experiment.dataset_builder = UcidPublic()
    experiment.load_from(path)

    experiment.evaluate(True)


test_twostream()
