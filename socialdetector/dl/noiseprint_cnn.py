from tensorflow.python.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout

from socialdetector.dl.jpeg_cnn import MyCNNJpeg
from socialdetector.dl.model import GenericModel, MultiModel, CombineModel, StreamModel


class NoiseprintModel(StreamModel):
    classes = 3

    def get_stream_model(self, input_img):
        layer = input_img
        layer = Conv2D(16, (3, 3), activation=self.conv_activation, padding=self.padding)(layer)
        layer = MaxPooling2D((2, 2))(layer)
        layer = BatchNormalization()(layer)
        layer = Conv2D(32, (3, 3), activation=self.conv_activation, padding=self.padding)(layer)
        layer = MaxPooling2D((2, 2))(layer)
        layer = BatchNormalization()(layer)
        layer = Conv2D(64, (3, 3), activation=self.conv_activation, padding=self.padding)(layer)
        layer = MaxPooling2D((2, 2))(layer)

        """layer = Conv2D(32, (3, 3), activation=self.conv_activation, padding=self.padding)(layer)
        layer = BatchNormalization()(layer)
        layer = MaxPooling2D((2, 2))(layer)"""
        return Flatten()(layer)

    def get_output_model(self, input_img):
        layer = input_img
        layer = Dense(128, activation=self.activation)(layer)
        layer = Dropout(0.5)(layer)
        layer = Dense(128, activation=self.activation)(layer)
        layer = Dropout(0.5)(layer)
        layer = Dense(128, activation=self.activation)(layer)
        layer = Dense(self.classes, activation='softmax')(layer)
        return layer

    padding = 'valid'
    conv_activation = 'relu'
    activation = 'swish'

    def get_input_shape(self):
        return 64, 64, 1


class FullModel(CombineModel):
    classes = 3

    activation = "swish"

    def get_output_model(self, layer):
        # layer = Dropout(0.2)(layer)
        layer = Dense(256, activation=self.activation)(layer)
        layer = Dropout(0.5)(layer)
        layer = Dense(256, activation=self.activation)(layer)
        layer = Dropout(0.5)(layer)
        layer = Dense(256, activation=self.activation)(layer)
        layer = Dense(self.classes, activation='softmax')(layer)
        return layer

    def __init__(self):
        super().__init__([MyCNNJpeg(), NoiseprintModel()])
