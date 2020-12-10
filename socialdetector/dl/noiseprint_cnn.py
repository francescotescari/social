from tensorflow.python.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout

from socialdetector.dl.model import GenericModel


class NoiseprintModel(GenericModel):
    padding = 'valid'
    conv_activation = 'relu'
    activation = 'swish'


    def model_structure(self, input_img):
        layer = input_img
        layer = Conv2D(100, (3, 3), activation=self.conv_activation, padding=self.padding)(layer)
        layer = BatchNormalization()(layer)
        layer = Conv2D(100, (3, 3), activation=self.conv_activation, padding=self.padding)(layer)
        layer = MaxPooling2D((2, 2))(layer)
        layer = BatchNormalization()(layer)
        layer = Conv2D(100, (3, 3), activation=self.conv_activation, padding=self.padding)(layer)
        layer = BatchNormalization()(layer)
        layer = Conv2D(100, (3, 3), activation=self.conv_activation, padding=self.padding)(layer)
        layer = MaxPooling2D((2, 2))(layer)
        layer = Conv2D(10, (3, 3), activation=self.conv_activation, padding=self.padding)(layer)
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
        return 64, 64, 1