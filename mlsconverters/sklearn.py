from mls.models import Algorithm, HyperParameter, HyperParameterSetting, Implementation, Run
from rdflib import Literal
import sklearn
import json

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
                    d[k] = xsd_type(v)
                except:
                    d[k] = xsd_type(str(v))
        return d

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
