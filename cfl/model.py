from keras import layers
import keras


def load_model():
    model = keras.Sequential(
        [
            layers.Input(shape=(30,)),          # 30 features
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"), 
        ]
    )

    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.02),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model