from mls.models import Algorithm, HyperParameter, HyperParameterSetting, Implementation, Run
import sklearn
import json
import numpy as np
from scipy.stats._distn_infrastructure import rv_frozen
from functools import partial


# TODO why is this unused?
def xsd_type(v):
    xsd_type = "xsd:anyURI"
    if type(v) == bool:
        xsd_type = "xsd:boolean"
    elif type(v) == int:
        xsd_type = "xsd:int"
    elif type(v) == float:
        xsd_type = "xsd:float"
    elif type(v) == str:
        xsd_type = "xsd:string"
    return {'@type': xsd_type, '@value': v}


def normalize_float(v):
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return str(v)
    else:
        return v


def standardize_types(v):
    if isinstance(v, np.ndarray):
        return [normalize_float(x) for x in v.tolist()]
    elif isinstance(v, float):
        return normalize_float(v)
    elif callable(v):
        return str(v) # TODO
    elif isinstance(v, rv_frozen):
        return {'dist_name': v.dist.name, 'args': v.args, 'kwds': v.kwds}
    return v


def deep_get_params(value, custom_type_handler=lambda x: x):
    recurse = partial(deep_get_params, custom_type_handler=custom_type_handler)

    value = custom_type_handler(value)
    value = standardize_types(value)

    if isinstance(value, (list, tuple)):
       return [recurse(x) for x in value]
    elif isinstance(value, dict):
       return {k: recurse(v) for k, v in value.items()}
    else:
        try:
            p = value.get_params()
            t = type(value).__module__ + '.' + type(value).__name__
            return {'@value': {'type': t, 'params': recurse(p)}}
        except AttributeError:
            try:
                json.dumps(value)
                return value
            except TypeError as e:
                raise NotImplementedError("can't convert model or param of type {} to mls: {}".format(type(value), e))


def to_mls(sklearn_model: sklearn.base.BaseEstimator):
    params = sklearn_model.get_params()

    def blank_node(id):
        return "_:{}".format(id)

    params = deep_get_params(sklearn_model)

    model_class = "_:{}.{}".format(type(sklearn_model).__module__, type(sklearn_model).__name__)

    model = Run()

    implementation = Implementation(id=model_class)
    implementation.parameters = [HyperParameter(id=blank_node(key)) for key in params.keys()]
    implementation.implements = Algorithm(id=model_class)
    model.executes = implementation

    model.input_values = [
        HyperParameterSetting(value=val, specified_by={"@id": blank_node(key)})
        for key, val in params.items() if val is not None
    ]

    return model.asjsonld()
