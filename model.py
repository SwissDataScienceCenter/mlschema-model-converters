from renku.core.models import mls
from rdflib import Literal
import sklearn

def to_mls(sklearn_model: sklearn.base.BaseEstimator):
    params = sklearn_model.get_params()
    model_class = type(sklearn_model).__module__ + '.' + type(sklearn_model).__name__

    model = mls.Run()

    implementation = mls.Implementation(id=model_class)
    implementation.parameters = [mls.HyperParameter(id=key) for key in params.keys()]
    implementation.implements = mls.Algorithm(id=model_class)
    model.executes = implementation

    model.input_values = [
        mls.HyperParameterSetting(value=val, specified_by=key)
        for key, val in params.items() if val is not None
    ]

    return model.asjsonld()
