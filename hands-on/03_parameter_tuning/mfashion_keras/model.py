import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation
from . import callback

NUM_CLASSES=10

def preprocess(feature, label=None):
    feature = tf.expand_dims(feature, axis=-1)
    feature = tf.divide(tf.cast(feature, tf.float32), 255.0)
    if label is not None:
        label = tf.one_hot(label, NUM_CLASSES)
    return feature, label

def dnn_model(hparams):
    model = tf.keras.Sequential([
      Flatten(),
      Dense(units = 128, activation = tf.nn.relu),
      Dense(units = 64, activation = tf.nn.relu),
      Dense(units = 32, activation = tf.nn.relu),
      Dropout(rate = hparams["dropout_rate"]),
      Dense(10, activation = 'softmax', name='output'),
    ])
    return model

def cnn_model(hparams):
    model = tf.keras.Sequential()
    model.add(Conv2D(hparams["filter_size_1"], kernel_size = hparams["kernel_size"], activation = "relu", input_shape=(28, 28, 1), name="image"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(hparams["dropout_rate"]))
    model.add(Conv2D(hparams["filter_size_2"], kernel_size = hparams["kernel_size"], activation = "relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(hparams["dropout_rate"]))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    if hparams["batch_norm"]:
      model.add(BatchNormalization())
      model.add(Activation(activation = tf.nn.relu))
    model.add(Dropout(hparams["dropout_rate_2"]))
    model.add(Dense(10, activation = 'softmax', name='output'))
    return model

def train_and_evaluate(output_dir, hparams):
    (X_train, y_train), _ = fashion_mnist.load_data()
    X_train = np.expand_dims(X_train, -1)
    X_train = X_train / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)

    tensorboard_callback   = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(output_dir, "logs"))
    earlystopping_callback = ## <TODO> EarlyStoppingを追加しよう

    if hparams["model"] == "dnn":
      model = dnn_model(hparams)
    else:
      model = cnn_model(hparams)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hparams["_____"]), # <TODO> learning_rateをparmeterから受け取れるようにしよう
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.CategoricalCrossentropy()]
    )
    model.fit(
        X_train, y_train,
        batch_size=hparams["batch_size"],
        epochs=hparams["train_steps"],
        validation_split=0.1,
        callbacks=[tensorboard_callback, earlystopping_callback, callback.MyMetricCallback()]
    )
    model.save(output_dir, save_format="tf")
