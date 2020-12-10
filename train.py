from tensorflow.python.keras.optimizer_v2.adam import Adam

from data_loader import *
from socialdetector.dataset_generator import encode_coefficients_my
from socialdetector.dl.jpeg_cnn import PaperCNNModel, MyCNNJpeg
from socialdetector.dl.model import GenericModel
from socialdetector.dl.noiseprint_cnn import NoiseprintModel

steps_per_epoch = 6690//2

train, validation, test = ucid_social(noiseprint=True, batch=[256, 256, 0])

load_path = None
#load_path = "./history/model_1607438398.047033/08-0.0339-0.0843.h5"

if load_path is None:
    #m = PaperCNNModel()
    #m = MyCNNJpeg()
    m = NoiseprintModel()
    m.build_model()
    m.compile(loss_function='categorical_crossentropy', metric_functions=['accuracy'], optimizer=Adam(lr=0.001))
else:
    m = GenericModel.load_from(load_path)
m.register_std_callbacks(tensorboard_logs_folder='./tsboard', checkpoint_path='./history')
# m.registered_callbacks.append(PredictCallback('./validate'))
m.train_with_generator(train.repeat(), epochs=20000, steps_per_epoch=steps_per_epoch, validation_data=validation)
