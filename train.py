import socialdetector.tf_options_setter
import tensorflow as tf
from socialdetector.dataset_generator import my_data_generator
from socialdetector.dl.jpeg_cnn import PaperCNNModel
from socialdetector.dl.model import GenericModel

batch_size = 128
steps_per_epoch = 2000

train, validate, test = my_data_generator(batch_size=batch_size, validate=500, seed=0)
train = train.prefetch(tf.data.experimental.AUTOTUNE).repeat()
validate = validate.cache()
test = test.cache()


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
m.train_with_generator(train, epochs=20000, steps_per_epoch=steps_per_epoch, validation_data=validate)
