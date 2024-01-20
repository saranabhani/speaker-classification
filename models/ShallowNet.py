from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense


class ShallowNet:
    @staticmethod
    def build(width, height, depth, classes, dropout=0.5):
        model = Sequential()
        input_shape = (height, width, depth)

        model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
        
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model
