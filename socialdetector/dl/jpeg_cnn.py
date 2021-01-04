from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout, BatchNormalization, Conv2D, \
    MaxPooling2D, GaussianDropout, GaussianNoise
from tensorflow.python.keras.regularizers import l2
from tf_siren import Sine

from socialdetector.dl.model import GenericModel, StreamModel


class PaperCNNModel(GenericModel):
    conv_activation = 'relu'
    activation = 'swish'
    classes = 3

    def model_structure(self, input_img):
        layer = input_img
        layer = Conv1D(100, 3, activation=self.conv_activation)(layer)
        layer = BatchNormalization()(layer)
        layer = MaxPooling1D()(layer)
        layer = Conv1D(100, 3, activation=self.conv_activation)(layer)
        layer = BatchNormalization()(layer)
        layer = MaxPooling1D()(layer)
        #layer = Conv1D(4, 3, activation=self.conv_activation)(layer)
        # layer = BatchNormalization()(layer)
        # layer = MaxPooling1D()(layer)
        layer = Flatten()(layer)
        # layer = Dropout(0.5)(layer)
        layer = Dense(256, activation=self.activation)(layer)
        layer = Dropout(0.5)(layer)
        layer = Dense(256, activation=self.activation)(layer)
        layer = Dropout(0.5)(layer)
        layer = Dense(256, activation=self.activation)(layer)
        layer = Dense(self.classes, activation='softmax')(layer)
        return layer

    def get_input_shape(self):
        return 909, 1


class MyCNNJpeg(StreamModel):
    classes = 3

    def get_stream_model(self, input_img):
        layer = input_img
        layer = Conv2D(64, (3, 3), activation=self.conv_activation, padding=self.padding,
                       data_format='channels_first')(layer)
        layer = BatchNormalization()(layer)
        layer = Conv2D(64, (3, 3), activation=self.conv_activation, padding=self.padding,
                       data_format='channels_first')(layer)
        # layer = Sine()(layer)
        layer = BatchNormalization()(layer)
        layer = MaxPooling2D((2, 2), data_format='channels_first')(layer)
        layer = Conv2D(128, (3, 3), activation=self.conv_activation, padding=self.padding,
                       data_format='channels_first')(layer)

        #layer = MaxPooling2D((2, 1), data_format='channels_first')(layer)
        # layer = Conv2D(256, (3, 3), activation='sigmoid', padding='valid',data_format='channels_first')(layer)

        layer = Flatten(name='flat_dct')(layer)
        return layer

    def get_output_model(self, input_img):
        layer = input_img
        # layer = Dropout(0.5)(layer)
        layer = Dense(128, activation=self.activation)(layer)
        layer = Dropout(0.5)(layer)
        layer = Dense(128, activation=self.activation)(layer)
        layer = Dropout(0.5)(layer)
        layer = Dense(128, activation=self.activation)(layer)
        layer = Dense(self.classes, activation='softmax')(layer)
        return layer

    conv_activation = 'relu'
    activation = 'swish'
    padding = 'same'

    def get_input_shape(self):
        return 9, 20, 11
