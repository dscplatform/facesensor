import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Activation, Dropout, Flatten, MaxPooling2D
from keras.optimizers import Adam

def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation="relu", input_shape=(1,112,92), data_format="channels_first"))
    model.add(Conv2D(32, (3,2), activation="relu", data_format="channels_first"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, (3,3), activation="relu", data_format="channels_first"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.10))
    model.add(Dense(40, activation="softmax"))
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])
    return model
