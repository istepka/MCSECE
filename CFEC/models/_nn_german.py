from typing import Sequence
from evaluation.data import GermanData
from keras.models import Sequential
from keras.layers import Dense, Input, Concatenate


def create_model():
    # create model
    model = Sequential([
        Dense(2, activation='softmax', input_shape=(61,))
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    german_data = GermanData('evaluation/data/datasets/input_german.csv', 'evaluation/data/datasets/labels_german.csv')
    model = create_model()
    model.summary()
    model.fit(german_data.X_train, german_data.y_train, epochs=30)
    scores = model.evaluate(german_data.X_test, german_data.y_test, verbose=0)
    print(scores)
    model.save('evaluation/models/model_german')
