import re
import time

import tensorflow as tf
import tensorflow_datasets
from tensorflow_datasets.core import benchmark

from socialdetector.dataset_generator import my_data_generator

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

from predict import PredictCallback, file_dataset
from socialdetector.datagenerator import *
from socialdetector.dl.jpeg_cnn import PaperCNNModel
from socialdetector.dl.model import GenericModel
from socialdetector.utility import FileWalker

fb_folders = ["C:\\Users\\franz\\Downloads\\Datasets\\ucid_social\\ucid\\facebook", ]
tw_folders = ["C:\\Users\\franz\\Downloads\\Datasets\\ucid_social\\ucid\\twitter", ]
fl_folders = ["C:\\Users\\franz\\Downloads\\Datasets\\ucid_social\\ucid\\flickr", ]

jpeg_filter_fn = lambda x: x.endswith(".jpg") or x.endswith(".jpeg")


def generator(folders, label, filter_fn):
    walkers = [FileWalker(folder, filter_fn) for folder in folders]
    gen = [PathTupleGenerator(walker,
                              {
                                  'noiseprint_path': "C:\\Users\\franz\\Downloads\\Datasets\\ucid_social\\ucid_noiseprint"
                              },
                              origin_dir="C:\\Users\\franz\\Downloads\\Datasets\\ucid_social\\ucid"
                              ) for walker in walkers]
    gen = ConcatDataGenerator(gen)
    gen = ShuffleDataGenerator(gen, 100000)
    # gen = LogDataGenerator(gen, source_key=None, destination_key=None)
    # gen = NoiseprintDataGenerator(gen, source_key='noiseprint_path', destination_key='noiseprint')
    gen = path_to_cnn_generator(gen)
    gen = ShuffleDataGenerator(gen, 1000)
    gen = AppendDataDataGenerator(gen, label, 'label')
    return gen


def all_gen(folder_label_list, filter_fn):
    gen = HomogeneousPollingDataGenerator([generator(fl[0], fl[1], filter_fn=filter_fn) for fl in folder_label_list])
    gen = InputOutputDataGenerator(gen, input_key='input', output_key='label')
    return gen


def dataset_generator(folder_label_list, filter_fn, train_val_test_filters):
    if [_ for _ in train_val_test_filters].count(None) > 1:
        raise ValueError()

    def build_strong_filter_fn(all_fns, true_one):
        def check(x):
            for i, fn in enumerate(all_fns):
                if fn is None:
                    continue
                res = fn(x)
                if (i == true_one and not res) or (i != true_one and res):
                    return False
            return True

        return check

    filter_fns = [build_strong_filter_fn(train_val_test_filters, i) for i in range(len(train_val_test_filters))]

    def gen_fun(i, ds_filter):
        def _():
            print("Resetting generator %i" % i)
            return all_gen(folder_label_list, lambda x: (filter_fn(x) and ds_filter(x)))
        return _

    generators = [gen_fun(i, ds_filter) for i, ds_filter in enumerate(filter_fns)]
    return tuple(
        Dataset.from_generator(gen, output_types=(tf.float32, tf.int32), output_shapes=((909, 1), (3,))) for gen in
        generators)


def is_dataset_file(k, m):
    def h(s: str):
        return sum([ord(c) ** 2 for c in s])

    if isinstance(k, int):
        def check(x):
            return h(x) % m == k
    elif callable(k):
        def check(x):
            return k(h(x) % m)
    else:
        raise ValueError()
    return check

train, validate, test = dataset_generator((
    (fb_folders, np.array([1, 0, 0])),
    (tw_folders, np.array([0, 1, 0])),
    (fl_folders, np.array([0, 0, 1]))
), jpeg_filter_fn, (
    None,
    is_dataset_file(lambda x: x<10, 128),
    is_dataset_file(10, 128),
))

train = train.repeat().batch(128).prefetch(tf.data.experimental.AUTOTUNE)
validate = validate.batch(64).cache()
test = test.batch(32)



load_path = None
# load_path = ".\\history\\model_1606777605.1109207\\18-0.0766-1.5952.h5"

if load_path is None:
    m = PaperCNNModel()
    m.build_model()
    m.compile(loss_function='categorical_crossentropy', metric_functions=['accuracy'])
else:
    m = GenericModel.load_from(load_path)
m.register_std_callbacks(tensorboard_logs_folder='./tsboard', checkpoint_path='./history')
# m.registered_callbacks.append(PredictCallback('./validate'))
m.train_with_generator(train, epochs=2000, steps_per_epoch=2000, validation_data=validate)
