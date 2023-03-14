import tensorflow as tf
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from typing import List

from dataset_transform import imbalance_aware_split

print(str(tf.config.list_physical_devices()))
tf.config.run_functions_eagerly(True)


def train_model(dataset_shortname: str, 
                seed: int = 44, 
                test_size_per_class: int = 100,
                layers: List[int] = [128, 64],
                ) -> None:
    '''
    `dataset_shortname`: shortname for dataset e.g. 'adult', 'fico', 'german'
    `seed`: seed for random number generator
    `test_size_per_class`: number of instances that each class should have in the test set
    '''
    seed = 44
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Get configs
    with open(f'data/{dataset_shortname}_constraints.json', 'r') as f:
        constr = json.load(f)

    # Load one-hot-encoded data
    dataset = pd.read_csv(f"data/{dataset_shortname}_train_ohe.csv")
    target_feature_col = constr['target_feature']

    #X_train, X_test, Y_train, Y_test = train_test_split(dataset[constr['features_order_after_split']], dataset['income'], test_size=0.2, random_state=seed)
    train, test = imbalance_aware_split(
        dataset[constr['features_order_after_split'] + [target_feature_col]], 
        target_feature_col=constr['target_feature'], 
        test_size_per_class=test_size_per_class,
        )

    X_train = train.drop([target_feature_col], axis=1)
    X_test = test.drop([target_feature_col], axis=1) 
    Y_train = train[target_feature_col]
    Y_test = test[target_feature_col] 
    
    # To numpy and split target to one-hot-encoded binary 
    X_train = X_train.to_numpy()
    Y_train = pd.get_dummies(Y_train).to_numpy()
    X_test = X_test.to_numpy()
    Y_test = pd.get_dummies(Y_test).to_numpy()

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    class_weights = np.round((1 / Y_train.sum(axis=0)) * (Y_train.sum() / 2), 4) # from TF tutorial on class weights
    tf_class_weights = {0: class_weights[0], 1: class_weights[1]}

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input((constr['features_count_split_without_target'],)))
    
    for layer_neurons in layers:
        model.add(tf.keras.layers.Dense(layer_neurons))
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(2))
    model.add(tf.keras.layers.Softmax())


    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(),
            tf.keras.metrics.Recall(),
            ]
    )
    with tf.device("/GPU:0"):
        model.fit(
            X_train, 
            Y_train,
            epochs=50,
            batch_size=512,
            validation_data=(X_test, Y_test),
            shuffle=True,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10)
            ],
            verbose=1,
            class_weight=tf_class_weights
        )
    

    preds = np.argmax(model.predict(X_test), axis=1)
    preds_true = np.argmax(Y_test, axis=1)

    cm  = confusion_matrix(preds_true, preds)

    df_cm = pd.DataFrame(cm)

    sns.heatmap(df_cm, annot=True)
    plt.show()

    model.save(f'models/{dataset_shortname}_NN.h5', overwrite=True, save_format='h5')
    model.save(f'models/{dataset_shortname}_NN', overwrite=True)

    print(f'Models saved to "models/{dataset_shortname}_NN"')

if __name__ == '__main__':
    #train_model('german', test_size_per_class=100, layers=[128, 64]])
    #train_model('adult', test_size_per_class=500, layers=[128, 64])
    #train_model('fico', test_size_per_class=200, layers=[32, 16])
    train_model('compas', test_size_per_class=300, layers=[64, 32])