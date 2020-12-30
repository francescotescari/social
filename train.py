from tensorflow.python.keras.optimizer_v2.nadam import Nadam

from experiments import *
from socialdetector.dataset.social_images.social_images import UcidPublic, UcidSocial, IpLabThree, IpLabSeven

# path = "./history/noiseprintonly_1607693319/e08-l0.1334-b1.4151.h5"python -
path = None

dataset = UcidPublic()
dataset.download_and_prepare()
# UcidSocial().download_and_prepare()
# UcidPublic().download_and_prepare()
# IpLabThree().download_and_prepare()


experiment = ExpMyJpeg()
# name
experiment.extra = "long"
experiment.batch_size = 256
experiment.dataset_builder = dataset
experiment.optimizer = Nadam(0.0001)
# experiment.steps_per_epoch = 100


experiment.load_from(path).train()
