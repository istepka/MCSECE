import numpy as np

from evaluation.data import AdultData
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import f1_score, classification_report


def create_model(input_dim):
    model = Sequential()
    model.add(Dense(1, activation='sigmoid', input_shape=input_dim))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    """
    To train model, run this script from repository directory:
    python -m evaluation.models._nn_adult
    """
    adult_data = AdultData('evaluation/data/datasets/adult.csv')
    input_dim = adult_data.X_train.shape[1:]
    model = create_model(input_dim)
    X_train, y_train = adult_data.X_train, adult_data.y_train_binarized
    model.fit(X_train, y_train, epochs=10)
    scores = model.evaluate(adult_data.X_test, adult_data.y_test_binarized, verbose=0)
    print(scores)
    preds = model.predict(adult_data.X_test)
    print(classification_report(adult_data.y_test, np.argmax(preds, axis=-1)))
    model.save('evaluation/models/model_adult')
