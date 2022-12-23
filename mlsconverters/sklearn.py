import json

import numpy as np
import sklearn
from scipy.stats._distn_infrastructure import rv_frozen

from .common import generate_unique_id, normalize_float
from .models import (Algorithm, EvaluationMeasure, HyperParameter,
                     HyperParameterSetting, Implementation, ModelEvaluation,
                     Run, RunSchema)

EVALUATION_MEASURE_KEY = "evaluation_measure"


def evaluation_measure(func, value):
    if hasattr(func, "__qualname__"):
        _id = generate_unique_id("http://www.w3.org/ns/mls#ModelEvaluation")
        if func.__qualname__ == "accuracy_score":
            return ModelEvaluation(
                _id=_id,
                value=value,
                specified_by=EvaluationMeasure(_id="http://www.w3.org/ns/mls#accuracy"),
            )
        elif func.__qualname__ == "roc_auc_score":
            return ModelEvaluation(
                _id=_id,
                value=value,
                specified_by=EvaluationMeasure(_id="http://www.w3.org/ns/mls#auROC"),
            )
        elif func.__qualname__ == "f1_score":
            return ModelEvaluation(
                _id=_id,
                value=value,
                specified_by=EvaluationMeasure(_id="http://www.w3.org/ns/mls#F1"),
            )
        else:
            raise ValueError("unsupported evaluation measure")


def to_mls(sklearn_model: sklearn.base.BaseEstimator, **kwargs):
    params = sklearn_model.get_params()

    def standardize_types(v):
        if isinstance(v, np.ndarray):
            return [normalize_float(x) for x in v.tolist()]
        elif isinstance(v, float):
            return normalize_float(v)
        elif callable(v):
            return str(v)  # TODO
        elif isinstance(v, rv_frozen):
            return {"dist_name": v.dist.name, "args": v.args, "kwds": v.kwds}
        return v

    def deep_get_params(params):
        if isinstance(params, (list, tuple)):
            return [deep_get_params(x) for x in params]
        elif isinstance(params, dict):
            return {k: deep_get_params(v) for k, v in params.items()}
        else:
            v = standardize_types(params)
            try:
                p = v.get_params()
                t = type(v).__module__ + "." + type(v).__name__
                return {"@value": {"type": t, "params": deep_get_params(p)}}
            except AttributeError:
                try:
                    json.dumps(v)
                    return v
                except TypeError as e:
                    raise NotImplementedError(
                        "can't convert sklearn model of type {} to mls: {}".format(
                            type(sklearn_model), e
                        )
                    )

    params = deep_get_params(params)
    model_hash = sklearn_model.__hash__()
    model_class = "{}.{}".format(
        type(sklearn_model).__module__, type(sklearn_model).__name__
    )
    algo = Algorithm(_id=model_class)

    implementation = Implementation(
        _id=generate_unique_id("http://www.w3.org/ns/mls#Implementation"),
        parameters=[
            HyperParameter(key, model_hash=model_hash) for key in params.keys()
        ],
        implements=algo,
        version=sklearn.__version__,
    )

    input_values = [
        HyperParameterSetting(
            value=val,
            specified_by=HyperParameter(key, model_hash=model_hash),
            model_hash=model_hash,
        )
        for key, val in params.items()
        if val is not None
    ]

    output_values = []
    if EVALUATION_MEASURE_KEY in kwargs:
        eval_measure = kwargs[EVALUATION_MEASURE_KEY]
        output_values.append(evaluation_measure(eval_measure[0], eval_measure[1]))
    model = Run(model_hash, implementation, input_values, output_values, algo)
    return RunSchema().dumps(model)
