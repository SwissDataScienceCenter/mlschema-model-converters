from renku.core.models import mls
from rdflib import Literal
import sklearn
import json

def to_mls(sklearn_model: sklearn.base.BaseEstimator):
    params = sklearn_model.get_params()

    def deep_get_params(params):
        d = {}
        for k, v in params.items():
            try:
                p = v.get_params()
                t = type(v).__module__ + '.' + type(v).__name__
                d[k] = {'@value': {'type': t, 'params': deep_get_params(p)}}
            except AttributeError:
                try:
                    json.dumps(v)
                    d[k] = v
                except:
                    d[k] = str(v)
        return d

    params = deep_get_params(params)

    model_class = type(sklearn_model).__module__ + '.' + type(sklearn_model).__name__

    model = mls.Run()

    implementation = mls.Implementation(id=model_class)
    implementation.parameters = [mls.HyperParameter(id=key) for key in params.keys()]
    implementation.implements = mls.Algorithm(id=model_class)
    model.executes = implementation

    model.input_values = [
        mls.HyperParameterSetting(id='{}Setting'.format(key), value=val, specified_by=key)
        for key, val in params.items() if val is not None
    ]

    return model.asjsonld()
