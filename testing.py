from tensorflow.python.data import Dataset

import socialdetector.tf_options_setter
import tensorflow as tf
from socialdetector.dataset_generator import my_data_generator
from socialdetector.dl.jpeg_cnn import PaperCNNModel
from socialdetector.dl.model import GenericModel

batch_size = [128, 128, 0]
steps_per_epoch = 2000

train, validate, test = my_data_generator(batch_size=batch_size, validate=500, seed=0)
train = train.prefetch(tf.data.experimental.AUTOTUNE).repeat()
validate = validate.cache()
test = test.cache()

t1 = test.filter(lambda x, y: tf.reduce_all(tf.equal(y, [1, 0, 0]))).batch(128)
t2 = test.filter(lambda x, y: tf.reduce_all(tf.equal(y, [0, 1, 0]))).batch(128)
t3 = test.filter(lambda x, y: tf.reduce_all(tf.equal(y, [0, 0, 1]))).batch(128)

test = test.batch(128)

load_path = ".\\history\\model_1607342670.0880523\\40-0.0501-0.0531.h5"
load_path = "./history/model_1607342670.0880523/47-0.0483-0.0485.h5"

m = GenericModel.load_from(load_path)
m.register_std_callbacks(tensorboard_logs_folder='./tsboard', checkpoint_path='./history')
# m.registered_callbacks.append(PredictCallback('./validate'))
m.model.evaluate(t1)
m.model.evaluate(t2)
m.model.evaluate(t3)
m.model.evaluate(test)
