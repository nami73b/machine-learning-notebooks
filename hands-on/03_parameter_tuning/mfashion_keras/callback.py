import tensorflow as tf
import hypertune

hpt = hypertune.HyperTune()


class MyMetricCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        ## <TODO> ___に適切なものを入れてください
        ## ヒント: yamlで指定したmetricIdを確認してください
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag='___', 
            metric_value=logs['categorical_crossentropy'], 
            global_step=epoch
        )        
