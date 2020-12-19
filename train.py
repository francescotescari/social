import socialdetector.tf_options_setter
from data_loader import ucid_social, ucid_public, iplab_three
from experiments import *
from socialdetector.dataset_utils import str_endswith
from socialdetector.experiment import FilterSplitter

path = "./history/noiseprintonly_1607693319/e08-l0.1334-b1.4151.h5"
path = None

experiment = MyJpeg()
experiment.dataset_spec = ucid_public
experiment.repeat_train = True
experiment.ds_splitter = FilterSplitter(str_endswith("1\\.jpg"), str_endswith("2\\.jpg"))
"""val = experiment.get_datasets()[1]
print(next(val.as_numpy_iterator()))
print(next(val.as_numpy_iterator()))"""
experiment.load_from(path).train()


