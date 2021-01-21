from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.nadam import Nadam

from experiments import *
from socialdetector.dataset.social_images.social_images import UcidPublic, UcidSocial, IpLabThree, IpLabSeven

# path = "./history/noiseprintonly_1607693319/e08-l0.1334-b1.4151.h5"python -
path = None

dataset = IpLabThree()
dataset.download_and_prepare()
# UcidSocial().download_and_prepare()
# UcidPublic().download_and_prepare()
# IpLabThree().download_and_prepare()


experiment = ExpTwoStreams()
# name
experiment.extra = "std"
experiment.batch_size = 256
experiment.dataset_builder = dataset
experiment.shuffle = 150000
experiment.optimizer = Nadam(lr=0.0005)
# experiment.steps_per_epoch = 100


experiment.load_from(path).train()
