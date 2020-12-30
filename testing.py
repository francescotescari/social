import socialdetector.tf_options_setter
from experiments import ExpNoiseprint, ExpMyJpeg, ExpPaperJpeg
from socialdetector.dataset.social_images import UcidPublic, UcidSocial, IpLabSeven, IpLabThree


def test_noiseprintonly():
    path = "./history/noiseprint_only/ucid_public/1608137326/e10-l0.1924-v0.3477.h5"

    experiment = ExpNoiseprint()
    experiment.dataset_builder = UcidPublic()
    experiment.load_from(path).evaluate()


def test_myjpeg():
    path = "./history/my_jpeg/ucid_social/1608161156/e39-l0.0155-v0.0851.h5"
    experiment = ExpMyJpeg()
    experiment.dataset_builder = UcidSocial()
    experiment.load_from(path).evaluate()


def test_paperjpeg():
    path = "./history/jpeg_paper/1608074487/e39-l0.1786-v0.1852.h5"
    path = "./history/jpeg_paper/1608074487/e35-l0.1290-v0.2095.h5"
    experiment = ExpPaperJpeg()
    experiment.dataset_builder = UcidPublic()
    experiment.load_from(path).evaluate()


test_myjpeg()
