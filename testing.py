import os

from tensorflow.python.keras import Model
from tensorflow.python.keras.metrics import Recall

import socialdetector.tf_options_setter
from experiments import ExpNoiseprint, ExpMyJpeg, ExpPaperJpeg, ExpTwoStreams
from socialdetector.dataset.social_images import UcidPublic, UcidSocial, IpLabSeven, IpLabThree
from socialdetector.train_utils import EvaluateCallback


def test_noiseprintonly():
    path = r".\history\ip_lab_three\noiseprint_only_1302\1611240500.9153724_std\e21-l0.1581.h5"

    experiment = ExpNoiseprint()
    experiment.dataset_builder = IpLabThree()
    experiment.load_from(path).evaluate(True)


def test_myjpeg():
    path = r".\history\ip_lab_three\my_jpeg_1302\1611237324.2902849_std\e37-l0.0343.h5"
    experiment = ExpMyJpeg()
    experiment.dataset_builder = IpLabThree()
    experiment.load_from(path).evaluate(True)


def test_paperjpeg():
    path = r"./history\ucid_public\my_jpeg_1302\1610897119.8677504_std\e20-l0.1980-v0.4742.h5"
    path = r".\history\ucid_public\jpeg_paper_1302\1610995763.9480963_std\e11-l0.4775.h5"
    experiment = ExpPaperJpeg()
    experiment.dataset_builder = UcidPublic()
    experiment.load_from(path).evaluate(True)


def test_twostream():
    path = "./history/two_streams/ucid_public/1609540064.0625489_std/e230-l0.0450-v0.3101.h5"
    #path = ".\\history\\two_streams\\ucid_social\\1609585022.0416644_std\\e51-l0.0184-v0.0287.h5"
    path = r".\history\ip_lab_three\two_streams_12321\1611240931.5301056_std\e20-l0.0152.h5"
    #path = r".\history\ucid_social\two_streams_1302\1611054810.6574473_std\e03-l0.0948.h5"
    #path = r".\history\ip_lab_three\two_streams_1302\1611063645.5685453_std\e05-l0.0762.h5"
    experiment = ExpTwoStreams()
    experiment.dataset_builder = IpLabThree()
    experiment.load_from(path)

    experiment.evaluate(True)


test_twostream()
