from distutils.version import LooseVersion
from uuid import uuid1

import gorilla

from .common import fn_args_as_params, mls_add_param
from .io import log_renku_mls
from .models import (Algorithm, EvaluationMeasure, Implementation,
                     ModelEvaluation, Run, RunSchema)


def autolog():
    import keras

    class __MLSKerasCallback(keras.callbacks.Callback):
        def __init__(self):
            self.mls = Run(uuid1())

        def on_train_begin(self, logs=None):
            mls_add_param(self.mls, "num_layers", len(self.model.layers))
            mls_add_param(
                self.mls, "optimizer_name", type(self.model.optimizer).__name__
            )
            if hasattr(self.model.optimizer, "lr"):
                lr = (
                    self.model.optimizer.lr
                    if type(self.model.optimizer.lr) is float
                    else keras.backend.eval(self.model.optimizer.lr)
                )
                mls_add_param(self.mls, "learning_rate", lr)
            if hasattr(self.model.optimizer, "epsilon"):
                epsilon = (
                    self.model.optimizer.epsilon
                    if type(self.model.optimizer.epsilon) is float
                    else keras.backend.eval(self.model.optimizer.epsilon)
                )
                mls_add_param(self.mls, "epsilon", epsilon)

        def on_epoch_end(self, epoch, logs=None):
            if not logs:
                return
            # try_mlflow_log(mlflow.log_metrics, logs, step=epoch)

        def on_train_end(self, logs=None):
            return

        # As of Keras 2.4.0, Keras Callback implementations must define the following
        # methods indicating whether or not the callback overrides functions for
        # batch training/testing/inference
        def _implements_train_batch_hooks(self):
            return False

        def _implements_test_batch_hooks(self):
            return False

        def _implements_predict_batch_hooks(self):
            return False

    def _early_stop_check(callbacks):
        if LooseVersion(keras.__version__) < LooseVersion("2.3.0"):
            es_callback = keras.callbacks.EarlyStopping
        else:
            es_callback = keras.callbacks.EarlyStopping
        for callback in callbacks:
            if isinstance(callback, es_callback):
                return callback
        return None

    def _log_early_stop_callback_params(callback):
        if callback:
            try:
                earlystopping_params = {
                    "monitor": callback.monitor,
                    "min_delta": callback.min_delta,
                    "patience": callback.patience,
                    "baseline": callback.baseline,
                    "restore_best_weights": callback.restore_best_weights,
                }
                mls_add_params(self.mls, earlystopping_params)
            except Exception:  # pylint: disable=W0703
                return

    def _get_early_stop_callback_attrs(callback):
        try:
            return (
                callback.stopped_epoch,
                callback.restore_best_weights,
                callback.patience,
            )
        except Exception:  # pylint: disable=W0703
            return None

    def _run_and_log_function(
        self, original, args, kwargs, unlogged_params, callback_arg_index
    ):
        mls_callback = __MLSKerasCallback()
        model_class = "keras.Model"

        algo = Algorithm(_id="NeuralNetwork")
        params, input_values = fn_args_as_params(
            original, args, kwargs, mls_callback.mls._id, unlogged_params
        )
        mls_implementation = Implementation(
            model_class, params, algo, keras.__version__
        )

        mls_implementation.parameters += params
        mls_callback.mls.executes = mls_implementation
        mls_callback.mls.input_values += input_values

        early_stop_callback = None

        # Checking if the 'callback' argument of the function is set
        if len(args) > callback_arg_index:
            tmp_list = list(args)
            early_stop_callback = _early_stop_check(tmp_list[callback_arg_index])
            tmp_list[callback_arg_index] += [mls_callback]
            args = tuple(tmp_list)
        elif "callbacks" in kwargs:
            early_stop_callback = _early_stop_check(kwargs["callbacks"])
            kwargs["callbacks"] += [mls_callback]
        else:
            kwargs["callbacks"] = [mls_callback]

        _log_early_stop_callback_params(early_stop_callback)

        history = original(self, *args, **kwargs)

        log_renku_mls(
            RunSchema().dumps(mls_callback.mls), str(self.__hash__()), force=True
        )

        return history

    @gorilla.patch(keras.Model)
    def fit(self, *args, **kwargs):
        original = gorilla.get_original_attribute(keras.Model, "fit")
        unlogged_params = ["self", "x", "y", "callbacks", "validation_data", "verbose"]
        return _run_and_log_function(self, original, args, kwargs, unlogged_params, 5)

    @gorilla.patch(keras.Model)
    def fit_generator(self, *args, **kwargs):
        original = gorilla.get_original_attribute(keras.Model, "fit_generator")
        unlogged_params = [
            "self",
            "generator",
            "callbacks",
            "validation_data",
            "verbose",
        ]
        return _run_and_log_function(self, original, args, kwargs, unlogged_params, 4)

    settings = gorilla.Settings(allow_hit=True, store_hit=True)
    gorilla.apply(gorilla.Patch(keras.Model, "fit", fit, settings=settings))
    gorilla.apply(
        gorilla.Patch(keras.Model, "fit_generator", fit_generator, settings=settings)
    )
