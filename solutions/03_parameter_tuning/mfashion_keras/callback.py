import tensorflow as tf
import hypertune

hpt = hypertune.HyperTune()


class MyMetricCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag='ccentropy', 
            metric_value=logs['categorical_crossentropy'], 
            global_step=epoch
        )
