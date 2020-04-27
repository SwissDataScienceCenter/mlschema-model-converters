from mls.models import Algorithm, HyperParameter, HyperParameterSetting, Implementation, Run
import sklearn
import json
import numpy as np
from scipy.stats._distn_infrastructure import rv_frozen

def to_mls(sklearn_model: sklearn.base.BaseEstimator):
    params = sklearn_model.get_params()

    def blank_node(id):
        return "_:{}".format(id)

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

    def deep_get_params(params):
        if isinstance(params, (list, tuple)):
           return [deep_get_params(x) for x in params]
        elif isinstance(params, dict):
           return {k: deep_get_params(v) for k, v in params.items()}
        else:
            v = standardize_types(params)
            try:
                p = v.get_params()
                t = type(v).__module__ + '.' + type(v).__name__
                return {'@value': {'type': t, 'params': deep_get_params(p)}}
            except AttributeError:
                try:
                    json.dumps(v)
                    return v
                except TypeError as e:
                    raise NotImplementedError("can't convert sklearn model of type {} to mls: {}".format(type(sklearn_model), e))

    params = deep_get_params(params)

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
