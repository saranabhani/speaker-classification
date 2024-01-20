from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout, Reshape, LSTM

class CnnRnn:
    @staticmethod
    def build(width, height, depth, classes, dropout=0.5):
        input_shape = (height, width, depth)
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Reshape((-1,  256)))  # Reshape for RNN
        model.add(LSTM(256, return_sequences=False))
        model.add(Dense(2048, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(2048, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(classes, activation='softmax'))
        
        return model