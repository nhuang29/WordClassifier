import json
import numpy as np
from sklearn.model_selection import train_test_split
import keras as keras

DATA_PATH = "data.json"
LEARNING_RATE = 0.0001
EPOCHS = 40
BATCH_SIZE = 32
SAVED_MODEL_PATH = "../../server/flask/model.h5"
NUM_KEYWORDS = 10


def load_dataset(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)

    # extract inputs and targets
    X = np.array(data["MFCCs"])
    y = np.array(data["labels"])

    return X, y


def get_data_splits(data_path, test_size=0.1, test_validation=0.1):
    # load the dataset
    X, y = load_dataset(data_path)

    # create the train/validation/test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train,
                                                                    test_size=test_validation)

    # convert inputs from 2d to 3d arrays
    # "..." means to take all that already exists in the array
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape, learning_rate, error="sparse_categorical_crossentropy"):
    # build the network
    model = keras.Sequential()

    # conv layer 1
    model.add(keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=input_shape,
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))

    # conv layer 2
    model.add(keras.layers.Conv2D(32, (3, 3), activation="relu",
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))

    # conv layer 3
    model.add(keras.layers.Conv2D(32, (2, 2), activation="relu",
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding="same"))

    # flatten the output and feed it into a dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    # Dropout shuts down 30% of neurons during training randomly, so network is forced to adapt

    # another dense layer with a softmax classifier
    model.add(keras.layers.Dense(NUM_KEYWORDS, activation='softmax'))

    # compile the model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=error, metrics=["accuracy"])

    model.summary()

    return model


def main():
    # load train/validation/test data splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = get_data_splits(DATA_PATH)

    # build the CNN model

    # The input shape is going to be 3 dimensions (# of segments, # of coefficients, 1 (channel of image/RGB))
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape, LEARNING_RATE)

    # train the model
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_data=(X_validation, y_validation))

    # evaluate the model
    loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test error: {loss}, test accuracy: {test_accuracy}")

    # save the model
    model.save(SAVED_MODEL_PATH)


if __name__ == "__main__":
    main()