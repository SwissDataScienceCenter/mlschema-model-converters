import pytest
import numpy as np
np.random.seed(1337)

import keras
import keras.layers as layers

import mlsconverters.keras


@pytest.fixture
def random_train_data():
    return np.random.random((1000, 32))


@pytest.fixture
def random_one_hot_labels():
    n, n_class = (1000, 10)
    classes = np.random.randint(0, n_class, n)
    labels = np.zeros((n, n_class))
    labels[np.arange(n), classes] = 1
    return labels


def create_model():
    model = keras.Sequential()

    model.add(layers.Dense(64, activation='relu', input_shape=(32,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(lr=0.001, epsilon=1e-07),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


@pytest.mark.large
@pytest.mark.parametrize('fit_variant', ['fit', 'fit_generator'])
def test_keras_autolog(random_train_data, random_one_hot_labels, fit_variant):
    mlsconverters.keras.autolog()

    data = random_train_data
    labels = random_one_hot_labels

    model = create_model()

    if fit_variant == 'fit_generator':
        def generator():
            while True:
                yield data, labels
        model.fit_generator(generator(), epochs=10, steps_per_epoch=1)
    else:
        model.fit(data, labels, epochs=10)
