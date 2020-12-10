from tensorflow.python.data import Dataset

import socialdetector.tf_options_setter
import tensorflow as tf

from data_loader import ucid_social
from socialdetector.dataset_generator import encode_coefficients_my
from socialdetector.dl.model import GenericModel

train, validation, test = ucid_social(encoding=encode_coefficients_my)

t1 = test.filter(lambda x, y: tf.reduce_all(tf.equal(y, [1, 0, 0]))).batch(128)
t2 = test.filter(lambda x, y: tf.reduce_all(tf.equal(y, [0, 1, 0]))).batch(128)
t3 = test.filter(lambda x, y: tf.reduce_all(tf.equal(y, [0, 0, 1]))).batch(128)

b_test = test.batch(256)

load_path = ".\\history\\model_1607342670.0880523\\40-0.0501-0.0531.h5"
load_path = "./history/model_1607596343.5289435/33-0.0343-0.0494.h5"
#load_path = "./history/model_1607342670.0880523/135-0.0843-0.0447.h5"

m = GenericModel.load_from(load_path)
#m.register_std_callbacks(tensorboard_logs_folder='./tsboard', checkpoint_path='./history')
# m.registered_callbacks.append(PredictCallback('./validate'))
m.model.evaluate(t1)
m.model.evaluate(t2)
m.model.evaluate(t3)
m.model.evaluate(b_test)
