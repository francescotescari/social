from tensorflow.python.ops.string_ops import regex_full_match
from tensorflow_datasets.core import benchmark

from socialdetector.dataset_generator import encode_coefficients_paper
from socialdetector.dataset_utils import path_glob_dataset, tf_print
from socialdetector.experiment import DatasetSpec, AllTrainSplitter, DatasetConstructor
import os


class StdDsSpec(DatasetSpec):
    img_dir = "images"
    np_dir = "noiseprint"

    def __init__(self, ds_dir, rel_input_dirs, name, files_number=None, pixel_estimate=None):
        src_dir = os.path.join(ds_dir, self.img_dir)
        nps_dir = os.path.join(ds_dir, self.np_dir)
        path_datasets = [(path_glob_dataset(src_dir, dir_entry[0]), dir_entry[1]) for dir_entry in rel_input_dirs]
        super().__init__(path_datasets=path_datasets, files_number=files_number, pixel_estimate=pixel_estimate,
                         noiseprint_dir=nps_dir, origin_dir=src_dir)
        self.name = name

    def __repr__(self):
        return self.name


ucid_social = StdDsSpec(
    ds_dir="C:\\Users\\franz\\Downloads\\Datasets\\ucid_social",
    rel_input_dirs=(
        ("facebook\\**\\*.jpg", (1, 0, 0)),
        ("flickr\\**\\*.jpg", (0, 1, 0)),
        ("twitter\\**\\*.jpg", (0, 0, 1))
    ),
    files_number=10000,
    pixel_estimate=6689 * 256 * 64 * 64,
    name="ucid_social"
)

ucid_public = StdDsSpec(
    ds_dir="C:\\Users\\franz\\Downloads\\Datasets\\ucid_public",
    rel_input_dirs=(
        ("facebook\\**\\*.jpg", (1, 0, 0)),
        ("flickr\\**\\*.jpg", (0, 1, 0)),
        ("twitter\\**\\*.jpg", (0, 0, 1))
    ),
    files_number=1000,
    pixel_estimate=2866 * 256 * 64 * 64,
    name="ucid_public"
)

iplab_three = StdDsSpec(
    ds_dir="C:\\Users\\franz\\Downloads\\Datasets\\iplab",
    rel_input_dirs=(
        ("**\\*facebook*\\**\\*.jpg", (1, 0, 0)),
        ("**\\*flickr*\\**\\*.jpg", (0, 1, 0)),
        ("**\\*twitter*\\**\\*.jpg", (0, 0, 1))
    ),
    files_number=280,
    pixel_estimate=691 * 256 * 64 * 64,
    name="iplab_three"
)
iplab_three.path_datasets[1] = (iplab_three.path_datasets[1][0].filter(lambda x: not regex_full_match(x, ".*_old.*")),
                                iplab_three.path_datasets[1][1])  # Remove _old entries from flickr dataset


def bench_ds(dataset_spec: DatasetSpec, batch_size=256, encoding=encode_coefficients_paper):
    constructor = DatasetConstructor()
    constructor.dct_encoding = encoding
    constructor.batch_size = [batch_size, batch_size, batch_size]
    splitter = AllTrainSplitter()
    train, val, tst = constructor.get_split(dataset_spec, splitter)
    return benchmark(train)

# bench_ds(ucid_social, 256)

# print(count_batches(ucid_public, 256))
