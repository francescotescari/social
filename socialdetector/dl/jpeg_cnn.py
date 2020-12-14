from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout, BatchNormalization, Conv2D, \
    MaxPooling2D

from socialdetector.dl.model import GenericModel


class PaperCNNModel(GenericModel):
    conv_activation = 'relu'
    activation = 'relu'

    def model_structure(self, input_img):
        layer = input_img
        layer = Conv1D(100, 3, activation=self.conv_activation)(layer)
        #layer = BatchNormalization()(layer)
        layer = MaxPooling1D()(layer)
        layer = Conv1D(100, 3, activation=self.conv_activation)(layer)
        #layer = BatchNormalization()(layer)
        layer = MaxPooling1D()(layer)
        layer = Flatten()(layer)
        #layer = Dropout(0.125)(layer)
        layer = Dense(256, activation=self.activation)(layer)
        layer = Dropout(0.25)(layer)
        layer = Dense(256, activation=self.activation)(layer)
        layer = Dropout(0.25)(layer)
        layer = Dense(256, activation=self.activation)(layer)
        layer = Dense(3, activation='softmax')(layer)
        return layer

    def get_input_shape(self):
        return 909, 1


class MyCNNJpeg(GenericModel):
    conv_activation = 'relu'
    activation = 'swish'
    padding = 'same'

    def model_structure(self, input_img):
        layer = input_img
        layer = Conv2D(100, (3,3), activation=self.conv_activation, padding=self.padding, data_format='channels_first')(layer)
        layer = BatchNormalization()(layer)
        layer = Conv2D(100, (3, 3), activation=self.conv_activation, padding=self.padding, data_format='channels_first')(layer)
        layer = BatchNormalization()(layer)
        layer = MaxPooling2D((2,2), data_format='channels_first')(layer)
        layer = Conv2D(100, (3, 3), activation=self.conv_activation, padding=self.padding, data_format='channels_first')(layer)
        layer = BatchNormalization()(layer)
        layer = Conv2D(100, (3, 3), activation=self.conv_activation, padding=self.padding, data_format='channels_first')(layer)
        layer = BatchNormalization()(layer)
        layer = MaxPooling2D((2,2), data_format='channels_first')(layer)
        layer = Flatten()(layer)
        # layer = Dropout(0.125)(layer)
        layer = Dense(256, activation=self.activation)(layer)
        layer = Dropout(0.2)(layer)
        layer = Dense(256, activation=self.activation)(layer)
        layer = Dropout(0.2)(layer)
        layer = Dense(256, activation=self.activation)(layer)
        layer = Dense(3, activation='softmax')(layer)
        return layer

    def get_input_shape(self):
        return 9, 20, 11
